"""Show the FULL AR + TTA rollout pool on real, with GT overlaid.

Uses batched-JIT sampler for speed (ar_batched.sample_pool_one_clip).
Yellow = every rollout; green GT = covered; red GT = missed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import jax
import matplotlib.pyplot as plt
import numpy as np

from sim2real.data.canonicalize import CANONICAL_H, CANONICAL_W
from sim2real.eval_v2.coverage import (
    _chamfer_polylines, canonicalize_real_frame, gt_polyline_to_canonical,
    load_real_annotations,
)
from sim2real.model.unet_ar import (
    AttachmentHead, KnotGenerator, UNetARBackbone,
)
from sim2real.scripts.ar_batched import make_sampler, sample_pool_one_clip
from sim2real.scripts.eval_ar import load_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-draws", type=int, default=2)
    ap.add_argument("--n-attach", type=int, default=8)
    ap.add_argument("--n-rollouts", type=int, default=4)
    ap.add_argument("--score-thresh", type=float, default=0.02)
    ap.add_argument("--tta-angles", nargs="+", type=float,
                    default=[0.0, -8.0, 8.0])
    ap.add_argument("--coverage-thresh", type=float, default=12.0,
                    help="in CANONICAL 256 px for display")
    ap.add_argument("--n-clips", type=int, default=24)
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--roll-alpha", type=float, default=0.15)
    ap.add_argument("--preproc", choices=["canonical", "simlike"],
                    default="canonical",
                    help="canonical = sigma-scale + bandpass (v8-v16 default); "
                         "simlike = median-sub + /255 (v17+ trained on this scale)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params, cfg = load_ckpt(args.ckpt)
    backbone = UNetARBackbone(cfg=cfg)
    attach_head = AttachmentHead(cfg=cfg)
    knot_gen = KnotGenerator(cfg=cfg)
    sampler = make_sampler(cfg, backbone, attach_head, knot_gen)
    annots = load_real_annotations()[: args.n_clips]

    key = jax.random.key(0)
    rows = []
    import time
    from sim2real.eval_v2.simlike_preproc import simlike_canonicalize
    for ai, ann in enumerate(annots):
        if args.preproc == "canonical":
            try:
                canon, cfg_can = canonicalize_real_frame(ann["meta"],
                                                           ann["src_width_px"], T=cfg.T)
            except Exception:
                continue
            clip = canon["clip"]
            smed_native = canon["static_median"].astype(np.float32) / 255
            raw_middle = smed_native.copy()
            src_h, src_w = clip.shape[1], clip.shape[2]
            scale_y = cfg.H / src_h; scale_x = cfg.W / src_w
            clip_r = np.stack([cv2.resize(clip[t], (cfg.W, cfg.H),
                                            interpolation=cv2.INTER_AREA)
                                for t in range(clip.shape[0])], axis=0).astype(np.float32)
            smed_r = cv2.resize(smed_native, (cfg.W, cfg.H),
                                 interpolation=cv2.INTER_AREA).astype(np.float32)
            gt_canon = []
            for pl in ann["gt_polylines_native"]:
                g = gt_polyline_to_canonical(pl, ann["meta"], cfg_can,
                                               canonical_h=CANONICAL_H,
                                               canonical_w=CANONICAL_W)
                if len(g) >= 4:
                    gt_canon.append(g)
        else:  # simlike
            try:
                clip_r, smed_r, src_h, src_w, sy, sx = simlike_canonicalize(
                    ann["meta"], T=cfg.T, target_hw=(cfg.H, cfg.W))
            except Exception:
                continue
            # For display, use static-median resized to canonical-256 for consistent
            # scale with canonical viz.
            raw_middle = cv2.resize(smed_r, (CANONICAL_W, CANONICAL_H),
                                       interpolation=cv2.INTER_CUBIC)
            # GT in canonical-256 (matches historical viz coordinates)
            scale_y_disp = CANONICAL_H / src_h; scale_x_disp = CANONICAL_W / src_w
            gt_canon = []
            for pl in ann["gt_polylines_native"]:
                if len(pl) < 4: continue
                g = pl.astype(np.float32) * np.asarray([scale_y_disp, scale_x_disp])
                gt_canon.append(g)
            # rollouts will be produced in model-canvas coords → project to
            # canonical-256 via (H_model → H_canonical, W_model → W_canonical)
            scale_y = cfg.H / CANONICAL_H; scale_x = cfg.W / CANONICAL_W

        t0 = time.time()
        rollouts, key = sample_pool_one_clip(
            params, backbone, attach_head, knot_gen, cfg,
            clip_r, smed_r, args.tta_angles, flips=(False, True),
            n_draws=args.n_draws, n_attach=args.n_attach,
            n_rollouts=args.n_rollouts,
            score_thresh=args.score_thresh, key=key, _sampler=sampler)
        # convert to canonical 256 coords
        rollouts_canon = [r / np.asarray([scale_y, scale_x]) for r in rollouts]
        dt = time.time() - t0
        rows.append(dict(name=ann["name"], bg=raw_middle,
                          gt=gt_canon, rollouts=rollouts_canon))
        print(f"  [{ai+1}/{args.n_clips}]  {ann['name']}  "
              f"{len(rollouts_canon)} rollouts  {dt:.2f}s", flush=True)

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
        ax.imshow(gray, cmap="gray", extent=(0, CANONICAL_W - 1, CANONICAL_H - 1, 0))
        # 1. all rollouts as thin yellow (the pool)
        for rl in row["rollouts"]:
            ax.plot(rl[:, 1], rl[:, 0], "-", color="#ffd633",
                    linewidth=0.6, alpha=args.roll_alpha)
        n_hit = 0
        for gt in row["gt"]:
            hit = False
            best_rl = None
            if row["rollouts"]:
                dists = [_chamfer_polylines(rl, gt) for rl in row["rollouts"]]
                best_idx = int(np.argmin(dists))
                best_d = dists[best_idx]
                best_rl = row["rollouts"][best_idx]
                hit = best_d <= args.coverage_thresh
            # 2. best-per-GT rollout in cyan (thicker than the pool)
            if best_rl is not None:
                ax.plot(best_rl[:, 1], best_rl[:, 0], "-", color="#00e0ff",
                        linewidth=1.6, alpha=0.9)
            # 3. GT on top: green if any pool rollout covers, red otherwise
            color = "#33ff44" if hit else "#ff3333"
            ax.plot(gt[:, 1], gt[:, 0], "-", color=color, linewidth=2.4)
            if hit: n_hit += 1
        n_gt = len(row["gt"])
        ax.set_title(f"{row['name']}  {n_hit}/{n_gt}  ({len(row['rollouts'])} rollouts)",
                     fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, CANONICAL_W - 1); ax.set_ylim(CANONICAL_H - 1, 0)
    for i in range(len(rows), nrow * args.ncols):
        r, c = i // args.ncols, i % args.ncols
        axes[r][c].axis("off")

    fig.suptitle("AR + TTA — full rollout pool on real "
                 "(yellow = every rollout, cyan = best-per-GT, "
                 "green GT = covered, red GT = miss)",
                 fontsize=10)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
