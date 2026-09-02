"""Side-by-side: what the AR model sees on sim vs real.

Column layout: for each row we pair a sim clip (top T=4 frames, static_median)
with a real clip preprocessed the SAME way (sim-like: median-sub, /255).
Same intensity mapping, so any visible difference IS the domain gap.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2, jax, numpy as np, matplotlib.pyplot as plt

from sim2real.eval_v2.coverage import load_real_annotations
from sim2real.eval_v2.simlike_preproc import simlike_canonicalize
from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--n-pairs", type=int, default=6)
    ap.add_argument("--sim-seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sim_cfg = DiverseSimConfig(T=args.T, H=args.H, W=args.H,
                                 sigma_scale_residual=False)
    annots = load_real_annotations()[: args.n_pairs]

    rows = []
    sim_key = jax.random.key(args.sim_seed)
    for ann in annots:
        try:
            clip_r, smed_r, *_ = simlike_canonicalize(
                ann["meta"], T=args.T, target_hw=(args.H, args.H))
        except Exception:
            continue
        # Try up to 20 sim seeds to find one with alive flagellum
        for _ in range(20):
            sim_key, k = jax.random.split(sim_key)
            out = sample_clip(k, sim_cfg)
            if int(np.asarray(out["flagella"]["alive"]).sum()) >= 1:
                break
        sim_clip = np.asarray(out["clip_median"])
        sim_smed = np.asarray(out["temporal_median"])
        rows.append((ann["name"], sim_clip, sim_smed, clip_r, smed_r))

    # Layout: each row = 1 real annotation pair; columns = [sim BG, sim T frames..., real BG, real T frames...]
    ncols = 2 * (1 + args.T)     # smed + T
    fig, axes = plt.subplots(len(rows), ncols,
                              figsize=(ncols * 1.6, len(rows) * 1.8),
                              squeeze=False)
    for r, (name, sc, sm, rc, rm) in enumerate(rows):
        # Compute a common intensity mapping per-source (independent normalization).
        # We show the residual and static median with their OWN scale so we can
        # perceive the actual pixel structure.
        for col, (img, is_bg, tag) in enumerate([(sm, True, "sim BG")]
                                                  + [(sc[t], False, f"sim t={t}") for t in range(args.T)]
                                                  + [(rm, True, "real BG")]
                                                  + [(rc[t], False, f"real t={t}") for t in range(args.T)]):
            ax = axes[r][col]
            lo, hi = np.percentile(img, [1, 99])
            gray = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
            ax.imshow(gray, cmap="gray")
            # Row label at first col only
            title = f"{tag}\n[{img.min():.3f}, {img.max():.3f}]\nσ={img.std():.4f}"
            ax.set_title(title, fontsize=6)
            ax.set_xticks([]); ax.set_yticks([])
        # Row-side label
        axes[r][0].set_ylabel(name, fontsize=7)
    fig.suptitle("Model input: sim vs real (sim-like preproc). Each panel is\n"
                 "min-max normalized to [1, 99] percentiles for VIEWING, but the\n"
                 "raw σ shown in the title reflects actual model-input magnitude.",
                 fontsize=9)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
