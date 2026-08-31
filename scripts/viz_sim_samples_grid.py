"""Grid of sim samples (raw | median-subtracted) for visual inspection of diversity."""
from __future__ import annotations

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip

OUT = Path("/home/frans/sim2real_tracking/runs/sim_samples_grid.png")
N_SAMPLES = 16
H = W = 200


def rgb_gray(f):
    lo, hi = np.percentile(f, 1), np.percentile(f, 99)
    return np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1)


def rgb_signed(f, rng):
    return np.clip((f + rng) / (2 * rng), 0, 1)


def main():
    cfg = DiverseSimConfig(T=16, H=H, W=W, n_max_cells=3, n_max_flagella=6,
                           n_arc_pts=22)
    print("compiling sim...", flush=True)

    rows, cols = 4, 4  # 4x4 samples; each sample = (raw, sub) pair → 8 cols visual
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2 * 1.6, rows * 1.6),
                             squeeze=False,
                             gridspec_kw=dict(wspace=0.02, hspace=0.10))

    seeds = list(range(N_SAMPLES))
    for i, seed in enumerate(seeds):
        r, c = i // cols, i % cols
        out = sample_clip(jax.random.key(1000 + seed), cfg)
        raw = np.asarray(out["clip_raw"])
        med = np.asarray(out["clip_median"])
        # pick a mid frame; render with sequence-local color range
        t = raw.shape[0] // 2
        rng = float(np.percentile(np.abs(med), 99.5))
        rng = max(rng, 0.02)
        axes[r][2 * c].imshow(rgb_gray(raw[t]), cmap="gray")
        axes[r][2 * c + 1].imshow(rgb_signed(med[t], rng), cmap="seismic")

        # gather quick descriptors
        n_cells = int(out["cells"]["alive"].sum())
        n_flag = int(out["flagella"]["alive"].sum())
        n_dark = int((out["flagella"]["amp"] < 0).sum() & out["flagella"]["alive"].sum())
        pip_present = bool(out["pipette"]["present"])
        modes = np.asarray(out["flagella"]["mode"])
        alive = np.asarray(out["flagella"]["alive"])
        mode_names = ["κwave", "tip", "breast", "static", "lat"]
        modes_str = "/".join(mode_names[m] for m, a in zip(modes, alive) if a)[:24]

        axes[r][2 * c].set_title(
            f"[{seed}] cells={n_cells} flag={n_flag} pip={'y' if pip_present else 'n'}",
            fontsize=7.5)
        axes[r][2 * c + 1].set_title(f"modes: {modes_str or '—'}", fontsize=7)
        for c2 in (2 * c, 2 * c + 1):
            axes[r][c2].set_xticks([]); axes[r][c2].set_yticks([])
        print(f"  seed {seed}: cells={n_cells}  flag={n_flag}  pip={pip_present}  "
              f"modes={modes_str}  res.p99.5={rng:.3f}",
              flush=True)

    fig.suptitle(f"Sim samples ({N_SAMPLES} clips, H=W={H}). "
                 "Each pair: raw | median-subtracted (color = ±p99.5)",
                 fontsize=11)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
