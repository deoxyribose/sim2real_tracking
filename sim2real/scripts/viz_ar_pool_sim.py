"""Same layout as viz_ar_pool.py, but on fresh sim clips (v8 sim config)."""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from sim2real.eval_v2.coverage import _chamfer_polylines
from sim2real.model.unet_ar import (
    AttachmentHead, KnotGenerator, UNetARBackbone,
)
from sim2real.scripts.ar_batched import make_sampler, sample_pool_one_clip
from sim2real.scripts.eval_ar import load_ckpt
from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-draws", type=int, default=4)
    ap.add_argument("--n-attach", type=int, default=8)
    ap.add_argument("--n-rollouts", type=int, default=4)
    ap.add_argument("--score-thresh", type=float, default=0.02)
    ap.add_argument("--tta-angles", nargs="+", type=float,
                    default=[0.0, -5.0, 5.0, -10.0, 10.0, -15.0, 15.0])
    ap.add_argument("--coverage-thresh", type=float, default=6.0,
                    help="model canvas 128 px")
    ap.add_argument("--n-clips", type=int, default=24)
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--roll-alpha", type=float, default=0.10)
    ap.add_argument("--sigma-scale", action="store_true",
                    help="enable v9's σ-scale (v8 was trained WITHOUT this)")
    ap.add_argument("--sim-seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params, cfg = load_ckpt(args.ckpt)
    backbone = UNetARBackbone(cfg=cfg)
    attach_head = AttachmentHead(cfg=cfg)
    knot_gen = KnotGenerator(cfg=cfg)
    sampler = make_sampler(cfg, backbone, attach_head, knot_gen)

    sim_cfg = DiverseSimConfig(T=cfg.T, H=cfg.H, W=cfg.W,
                                 sigma_scale_residual=args.sigma_scale)

    key = jax.random.key(0)
    sim_key = jax.random.key(args.sim_seed)
    rows = []
    import time
    for ai in range(args.n_clips):
        sim_key, k = jax.random.split(sim_key)
        out = sample_clip(k, sim_cfg)
        clip = np.asarray(out["clip_median"])           # (T, H, W)
        smed = np.asarray(out["temporal_median"])       # (H, W)
        curves = np.asarray(out["curves"][cfg.T // 2])  # (N_flag, K, 2) (y, x)
        alive = np.asarray(out["flagella"]["alive"])
        gt = [curves[i] for i in range(curves.shape[0]) if bool(alive[i])]
        raw_middle = np.asarray(out["clip_raw"])[cfg.T // 2].astype(np.float32)

        t0 = time.time()
        rollouts, key = sample_pool_one_clip(
            params, backbone, attach_head, knot_gen, cfg,
            clip, smed, args.tta_angles, flips=(False, True),
            n_draws=args.n_draws, n_attach=args.n_attach,
            n_rollouts=args.n_rollouts,
            score_thresh=args.score_thresh, key=key, _sampler=sampler)
        rows.append(dict(idx=ai, bg=raw_middle, gt=gt, rollouts=rollouts))
        print(f"  [{ai+1}/{args.n_clips}]  {len(rollouts)} rollouts  "
              f"{len(gt)} gt  {time.time()-t0:.2f}s", flush=True)

    nrow = (len(rows) + args.ncols - 1) // args.ncols
    fig, axes = plt.subplots(nrow, args.ncols,
                              figsize=(args.ncols * 3.6, nrow * 3.4),
                              squeeze=False)
    for i, row in enumerate(rows):
        r, c = i // args.ncols, i % args.ncols
        ax = axes[r][c]
        bg = row["bg"]
        lo, hi = np.percentile(bg, [1, 99])
        gray = np.clip((bg - lo) / max(hi - lo, 1e-6), 0, 1)
        ax.imshow(gray, cmap="gray")
        for rl in row["rollouts"]:
            ax.plot(rl[:, 1], rl[:, 0], "-", color="#ffd633",
                    linewidth=0.6, alpha=args.roll_alpha)
        n_hit = 0
        for g in row["gt"]:
            if len(g) < 2: continue
            hit = False
            if row["rollouts"]:
                dists = [_chamfer_polylines(rl, g) for rl in row["rollouts"]]
                hit = min(dists) <= args.coverage_thresh
            color = "#33ff44" if hit else "#ff3333"
            ax.plot(g[:, 1], g[:, 0], "-", color=color, linewidth=2.4)
            if hit: n_hit += 1
        n_gt = len([g for g in row["gt"] if len(g) >= 2])
        ax.set_title(f"sim_{row['idx']:02d}  {n_hit}/{n_gt}  "
                     f"({len(row['rollouts'])} rollouts)", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, cfg.W - 1); ax.set_ylim(cfg.H - 1, 0)
    for i in range(len(rows), nrow * args.ncols):
        r, c = i // args.ncols, i % args.ncols
        axes[r][c].axis("off")

    fig.suptitle(f"V8 AR + wide TTA — sim rollouts "
                 f"(σ-scale={'on' if args.sigma_scale else 'off'}, "
                 f"yellow=every rollout, green GT=covered, red=miss)",
                 fontsize=10)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
