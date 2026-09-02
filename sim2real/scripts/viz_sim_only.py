"""Render sim clips WITHOUT the model — just BG + GT curves for debugging.

Fast, no ML, runs on CPU. Use to inspect sim rendering fixes."""
from __future__ import annotations
import argparse
from pathlib import Path

import jax, numpy as np, matplotlib.pyplot as plt

from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=24)
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sim_cfg = DiverseSimConfig(T=args.T, H=args.H, W=args.H,
                                 sigma_scale_residual=False)
    key = jax.random.key(args.seed)
    rows = []
    for i in range(args.n_clips):
        key, k = jax.random.split(key)
        out = sample_clip(k, sim_cfg)
        img = np.asarray(out["clip_raw"])[args.T // 2].astype(np.float32)
        curves = np.asarray(out["curves"])[args.T // 2]      # (N, K, 2)
        alive = np.asarray(out["flagella"]["alive"])         # (N,)
        pip_present = bool(np.asarray(out["pipette"]["present"]))
        rows.append((i, img, curves, alive, pip_present))

    nrow = (args.n_clips + args.ncols - 1) // args.ncols
    fig, axes = plt.subplots(nrow, args.ncols,
                              figsize=(args.ncols * 3.4, nrow * 3.4),
                              squeeze=False)
    for k, (i, img, curves, alive, pip) in enumerate(rows):
        r, c = k // args.ncols, k % args.ncols
        ax = axes[r][c]
        lo, hi = np.percentile(img, [1, 99])
        gray = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        ax.imshow(gray, cmap="gray")
        n_alive = 0
        for j in range(curves.shape[0]):
            g = curves[j]
            if bool(alive[j]):
                ax.plot(g[:, 1], g[:, 0], "-", color="#33ff44",
                        linewidth=1.6)
                ax.scatter([g[0, 1]], [g[0, 0]], color="#33ff44", s=8,
                            marker="o", edgecolors="black", linewidths=0.4)
                n_alive += 1
            else:
                ax.plot(g[:, 1], g[:, 0], ":", color="#ff3333",
                        linewidth=1.0, alpha=0.6)
        ax.set_title(f"sim_{i:02d}  {n_alive} alive  pip={pip}",
                     fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, args.H - 1); ax.set_ylim(args.H - 1, 0)

    fig.suptitle(f"Sim raw + GT curves (green=alive, red-dotted=OOF-killed)",
                 fontsize=10)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
