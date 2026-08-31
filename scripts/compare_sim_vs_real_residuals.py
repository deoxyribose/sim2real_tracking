"""Render sim clips (median-subtracted) next to real residuals, so we can
visually check whether the sim covers the real distribution.

Layout: 2 columns (raw | median-subtracted), rows = mix of real + sim.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image

from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip

ANNOT_MANIFEST = Path("/home/frans/sim2real_tracking/annotations/flagella_v0/manifest.json")
OUT = Path("/home/frans/sim2real_tracking/runs/sim_vs_real.png")

REAL_PICK = [
    "1912/uni_rotor/cell_5_20241219_171909",             # strong flagellum
    "CC124_cell2/Export_20211015_124646",                # medium flagellum
    "wetransfer_some-mediocre-examples_2025-01-14_1648/cell1_alive_20240513_171521",  # drift
    "CC124_cell1/2021_10_07/Export_20211007_174615",     # faint
]
N_REAL_FRAMES = 100
WINDOW = 21


def load_real(seq_name: str, n_frames: int = N_REAL_FRAMES):
    m = json.load(open(ANNOT_MANIFEST))
    entry = next(x for x in m if x["sequence"] == seq_name)
    src = entry["source"]
    ext = "." + src.rsplit(".", 1)[1]
    dirp = Path(src).parent
    ct, cb = int(entry["crop_top"]), int(entry["crop_bot"])
    files = sorted(glob.glob(str(dirp / f"*{ext}")))
    idxs = np.linspace(0, len(files) - 1, n_frames).astype(int)
    frames = []
    for i in idxs:
        f = files[i]
        img = tifffile.imread(f) if f.endswith(".tif") else np.array(Image.open(f))
        if img.ndim == 3:
            img = img.mean(-1)
        frames.append(img[ct:cb].astype(np.float32))
    stack = np.stack(frames)
    # normalize to [0, 1]
    return stack / 255.0


def sliding_median(stack, window):
    T = stack.shape[0]
    half = window // 2
    out = np.empty_like(stack)
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        out[t] = np.median(stack[lo:hi], axis=0)
    return out


def sample_sim(seed: int, T: int = 16, H: int = 200, W: int = 200):
    cfg = DiverseSimConfig(T=T, H=H, W=W, n_max_cells=3, n_max_flagella=6,
                           n_arc_pts=22)
    out = sample_clip(jax.random.key(seed), cfg)
    return np.asarray(out["clip_raw"]), np.asarray(out["clip_median"])


def rgb_gray(frame):
    lo, hi = np.percentile(frame, 1), np.percentile(frame, 99)
    g = np.clip((frame - lo) / max(hi - lo, 1e-6), 0, 1)
    return g


def rgb_signed(frame, rng):
    return np.clip((frame + rng) / (2 * rng), 0, 1)


def main():
    rows = []
    for seq in REAL_PICK:
        stack = load_real(seq)
        med = sliding_median(stack, WINDOW)
        res = stack - med
        t = stack.shape[0] // 2
        rng = float(np.percentile(np.abs(res), 99.5))
        rows.append(("real / " + seq.split("/")[-1][:28], stack[t], res[t], rng))

    # Sim samples: cover a spread of seeds
    print("compiling sim...", flush=True)
    for i, s in enumerate([1, 7, 12, 21, 33, 42, 55, 61]):
        raw, med = sample_sim(s)
        t = raw.shape[0] // 2
        rng = float(np.percentile(np.abs(med), 99.5))
        rows.append((f"sim / seed={s}", raw[t], med[t], rng))
        print(f"  sim seed {s}: raw range [{raw.min():.3f}, {raw.max():.3f}]  "
              f"residual p99.5={rng:.4f}", flush=True)

    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(6.2, 1.7 * n), squeeze=False)
    for r, (label, raw, res, rng) in enumerate(rows):
        axes[r][0].imshow(rgb_gray(raw), cmap="gray")
        axes[r][1].imshow(rgb_signed(res, rng), cmap="seismic")
        axes[r][0].set_ylabel(label, fontsize=7.5)
        for c in range(2):
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
        if r == 0:
            axes[r][0].set_title("raw", fontsize=9)
            axes[r][1].set_title("− median  (color range = ±p99.5)", fontsize=9)

    fig.suptitle("Real residuals (top) vs sim residuals (bottom)   —   "
                 "does the sim span the real distribution?", fontsize=10)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
