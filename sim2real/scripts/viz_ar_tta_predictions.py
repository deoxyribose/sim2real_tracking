"""Show AR + wide-TTA predictions on real, canonical raw background."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
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
from sim2real.scripts.eval_ar import load_ckpt
from sim2real.scripts.eval_ar_tta import run_augmented


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-draws", type=int, default=4)
    ap.add_argument("--n-attach", type=int, default=8)
    ap.add_argument("--n-rollouts", type=int, default=4)
    ap.add_argument("--score-thresh", type=float, default=0.02)
    ap.add_argument("--coverage-thresh", type=float, default=12.0,
                    help="in CANONICAL 256 px for display; 12 @ 256 = 6 @ 128")
    ap.add_argument("--tta-angles", nargs="+", type=float,
                    default=[0.0, -5.0, 5.0, -10.0, 10.0, -15.0, 15.0])
    ap.add_argument("--n-clips", type=int, default=24)
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params, cfg = load_ckpt(args.ckpt)
    backbone = UNetARBackbone(cfg=cfg)
    attach_head = AttachmentHead(cfg=cfg)
    knot_gen = KnotGenerator(cfg=cfg)
    annots = load_real_annotations()[: args.n_clips]

    key = jax.random.key(0)
    rows = []
    for ai, ann in enumerate(annots):
        try:
            canon, cfg_can = canonicalize_real_frame(ann["meta"],
                                                       ann["src_width_px"], T=cfg.T)
        except Exception:
            continue
        clip = canon["clip"]
        smed_native = canon["static_median"].astype(np.float32) / 255
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
                gt_canon.append(g * np.asarray([scale_y, scale_x]))

        all_rollouts = []
        for angle in args.tta_angles:
            for flipped in (False, True):
                r, key = run_augmented(params, backbone, attach_head, knot_gen, cfg,
                                          clip_r, smed_r, key,
                                          args.n_draws, args.n_attach,
                                          args.n_rollouts, args.score_thresh,
                                          angle, flipped)
                all_rollouts.extend(r)
        rows.append(dict(name=ann["name"], bg=smed_native,
                          gt=gt_canon, rollouts=all_rollouts,
                          scale_y=scale_y, scale_x=scale_x))
        print(f"  processed {ai+1}/{args.n_clips} ({len(all_rollouts)} rollouts)",
              flush=True)

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
        n_hit = 0
        for gt in row["gt"]:
            gt_canon = gt / np.asarray([row["scale_y"], row["scale_x"]])
            if not row["rollouts"]:
                continue
            dists = [_chamfer_polylines(rl, gt) for rl in row["rollouts"]]
            best_idx = int(np.argmin(dists))
            best_dist = dists[best_idx]
            hit = best_dist * (1.0 / row["scale_y"]) <= args.coverage_thresh
            if hit: n_hit += 1
            best = row["rollouts"][best_idx]
            best_canon = best / np.asarray([row["scale_y"], row["scale_x"]])
            color = "#33dd33" if hit else "#dd3333"
            ax.plot(best_canon[:, 1], best_canon[:, 0], "-", color="#ffcc00",
                    linewidth=1.8, alpha=0.9)
            ax.plot(gt_canon[:, 1], gt_canon[:, 0], "-", color=color, linewidth=2.0)
            ax.text(gt_canon[0, 1] + 3, gt_canon[0, 0] - 3,
                    f"d={best_dist / row['scale_y']:.1f}", fontsize=7, color=color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                              alpha=0.7))
        n_gt = len(row["gt"])
        ax.set_title(f"{row['name']}  {n_hit}/{n_gt} @ {args.coverage_thresh:.0f}px",
                     fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, CANONICAL_W - 1); ax.set_ylim(CANONICAL_H - 1, 0)

    for i in range(len(rows), nrow * args.ncols):
        r, c = i // args.ncols, i % args.ncols
        axes[r][c].axis("off")

    fig.suptitle("V8 AR + wide TTA: predictions on real (canonical bg), "
                 "yellow=closest rollout, green GT=hit, red GT=miss",
                 fontsize=10)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
