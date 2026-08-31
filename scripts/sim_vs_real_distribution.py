"""Overlay residual-value PDFs from sim (N clips) and real (16 sequences) to
verify the sim covers the real distribution (plus some, per the principle)."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image

from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip

ANNOT_MANIFEST = Path("/home/frans/sim2real_tracking/annotations/flagella_v0/manifest.json")
OUT = Path("/home/frans/sim2real_tracking/runs/sim_vs_real_distribution.png")

N_REAL_FRAMES = 100
N_SIM_CLIPS = 24
WINDOW = 21
H_SIM = W_SIM = 200


def load_real_residuals():
    m = json.load(open(ANNOT_MANIFEST))
    by_seq = {}
    for x in m:
        by_seq.setdefault(x["sequence"], x)
    seq_res = {}
    for name, entry in sorted(by_seq.items()):
        src = entry["source"]
        ext = "." + src.rsplit(".", 1)[1]
        dirp = Path(src).parent
        ct, cb = int(entry["crop_top"]), int(entry["crop_bot"])
        files = sorted(glob.glob(str(dirp / f"*{ext}")))
        idxs = np.linspace(0, len(files) - 1, N_REAL_FRAMES).astype(int)
        frames = []
        for i in idxs:
            f = files[i]
            img = tifffile.imread(f) if f.endswith(".tif") else np.array(Image.open(f))
            if img.ndim == 3:
                img = img.mean(-1)
            frames.append(img[ct:cb].astype(np.float32) / 255.0)
        stack = np.stack(frames)
        # sliding window median
        T = stack.shape[0]; half = WINDOW // 2
        med = np.empty_like(stack)
        for t in range(T):
            lo, hi = max(0, t - half), min(T, t + half + 1)
            med[t] = np.median(stack[lo:hi], axis=0)
        seq_res[name] = (stack - med).ravel()
        print(f"real {name.split('/')[-1][:32]:32s}  n={len(seq_res[name])}")
    return seq_res


def sample_sim_residuals():
    cfg = DiverseSimConfig(T=16, H=H_SIM, W=W_SIM, n_max_cells=3,
                           n_max_flagella=6, n_arc_pts=22)
    print(f"compiling sim (H=W={H_SIM}, T=16)...", flush=True)
    all_res = []
    for i in range(N_SIM_CLIPS):
        out = sample_clip(jax.random.key(1000 + i), cfg)
        all_res.append(np.asarray(out["clip_median"]).ravel())
        if i % 4 == 0:
            print(f"  sim clip {i+1}/{N_SIM_CLIPS}", flush=True)
    return np.concatenate(all_res)


def main():
    real = load_real_residuals()
    sim = sample_sim_residuals()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    edges = np.linspace(-0.6, 0.6, 241)

    # Per-sequence real, thin lines
    for name, vals in real.items():
        axes[0].hist(vals, bins=edges, density=True, histtype="step",
                     linewidth=0.9, alpha=0.55, color="#005ec4")
    # Overlaid: aggregate real + sim
    real_all = np.concatenate(list(real.values()))
    axes[0].hist(real_all, bins=edges, density=True, histtype="step",
                 linewidth=2.0, color="#005ec4", label="real (aggregate)")
    axes[0].hist(sim, bins=edges, density=True, histtype="step",
                 linewidth=2.0, color="#b02a2a", label="sim (aggregate)")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-4, 1e3)
    axes[0].set_xlabel("residual (median-subtracted, [0,1] intensity units)")
    axes[0].set_ylabel("density (log)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title(f"residual PDFs  —  {N_SIM_CLIPS} sim clips vs "
                      f"{len(real)} real seqs (thin blue)")
    axes[0].axvline(0, color="#000", lw=0.4)

    # Right: tail percentiles per source
    xs = np.array([50, 75, 90, 95, 99, 99.5, 99.9, 99.99])
    real_p = np.percentile(np.abs(real_all), xs)
    sim_p = np.percentile(np.abs(sim), xs)
    # x axis: −log10(1 − p/100) so p=99, 99.9, 99.99 space out linearly
    def _tx(p): return -np.log10(1.0 - p / 100.0 + 1e-12)
    for name, vals in real.items():
        p = np.percentile(np.abs(vals), xs)
        axes[1].plot(_tx(xs), p, "-", color="#005ec4", alpha=0.3, linewidth=0.7)
    axes[1].plot(_tx(xs), real_p, "o-", color="#005ec4", label="real (aggregate)")
    axes[1].plot(_tx(xs), sim_p, "s-", color="#b02a2a", label="sim (aggregate)")
    axes[1].set_xticks(_tx(xs))
    axes[1].set_xticklabels([f"p{x:g}" for x in xs], fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("percentile of |residual|")
    axes[1].set_ylabel("|residual|  ([0,1] units)")
    axes[1].set_title("tail coverage")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(True, which="both", alpha=0.2)

    fig.suptitle("Sim vs real residual distribution — verifying 'cover + some'",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")

    # Print a coverage summary
    print("\nCoverage summary (|residual| percentile: real vs sim):")
    for x, rp, sp in zip(xs, real_p, sim_p):
        marker = "✓" if sp >= rp else "✗ sim short"
        print(f"  p{x:6.2f}:  real={rp:.4f}  sim={sp:.4f}   {marker}")


if __name__ == "__main__":
    main()
