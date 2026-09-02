"""Compare Chamfer, ordered-knot-L2, and Hausdorff on the same rollout pool.

Runs the model on each real annotation and reports recall @ multiple
thresholds for each metric. Chamfer @ 6 px (canvas 128) is the number
we've been reporting; the others tell us if Chamfer is inflating it.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import cv2, jax, numpy as np

from sim2real.data.canonicalize import CANONICAL_H, CANONICAL_W
from sim2real.eval_v2.coverage import (
    _chamfer_polylines, canonicalize_real_frame, gt_polyline_to_canonical,
    load_real_annotations,
)
from sim2real.eval_v2.ordered_metric import ordered_l2, hausdorff
from sim2real.model.unet_ar import (
    AttachmentHead, KnotGenerator, UNetARBackbone,
)
from sim2real.scripts.ar_batched import make_sampler, sample_pool_one_clip
from sim2real.scripts.eval_ar import load_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-draws", type=int, default=4)
    ap.add_argument("--n-attach", type=int, default=16)
    ap.add_argument("--n-rollouts", type=int, default=4)
    ap.add_argument("--tta-angles", nargs="+", type=float,
                    default=[0.0, -5.0, 5.0, -10.0, 10.0, -15.0, 15.0])
    ap.add_argument("--k", type=int, default=25,
                    help="knots for resample in ordered_l2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params, cfg = load_ckpt(args.ckpt)
    backbone = UNetARBackbone(cfg=cfg)
    attach_head = AttachmentHead(cfg=cfg)
    knot_gen = KnotGenerator(cfg=cfg)
    sampler = make_sampler(cfg, backbone, attach_head, knot_gen)
    annots = load_real_annotations()

    key = jax.random.key(42)
    all_metrics = []      # list of (chamfer, ordered_l2, hausdorff) per GT
    for ai, ann in enumerate(annots):
        try:
            canon, cfg_can = canonicalize_real_frame(ann["meta"],
                                                       ann["src_width_px"], T=cfg.T)
        except Exception:
            continue
        clip = canon["clip"]
        smed = canon["static_median"].astype(np.float32) / 255
        src_h, src_w = clip.shape[1], clip.shape[2]
        sy, sx = cfg.H / src_h, cfg.W / src_w
        clip_r = np.stack([cv2.resize(clip[t], (cfg.W, cfg.H),
                                        interpolation=cv2.INTER_AREA)
                            for t in range(clip.shape[0])], axis=0).astype(np.float32)
        smed_r = cv2.resize(smed, (cfg.W, cfg.H),
                             interpolation=cv2.INTER_AREA).astype(np.float32)

        gts = []
        for pl in ann["gt_polylines_native"]:
            g = gt_polyline_to_canonical(pl, ann["meta"], cfg_can,
                                           canonical_h=CANONICAL_H,
                                           canonical_w=CANONICAL_W)
            if len(g) >= 4:
                gts.append(g * np.asarray([sy, sx]))
        if not gts: continue

        rollouts, key = sample_pool_one_clip(
            params, backbone, attach_head, knot_gen, cfg,
            clip_r, smed_r, args.tta_angles, flips=(False, True),
            n_draws=args.n_draws, n_attach=args.n_attach,
            n_rollouts=args.n_rollouts,
            score_thresh=0.02, key=key, _sampler=sampler)

        for g in gts:
            if not rollouts:
                all_metrics.append((float("inf"),) * 3); continue
            chamfers = [_chamfer_polylines(rl, g) for rl in rollouts]
            best_ch = min(range(len(chamfers)), key=lambda i: chamfers[i])
            ord_ls = [ordered_l2(rl, g, args.k) for rl in rollouts]
            best_ol = min(range(len(ord_ls)), key=lambda i: ord_ls[i])
            hauss = [hausdorff(rl, g) for rl in rollouts]
            best_h = min(range(len(hauss)), key=lambda i: hauss[i])
            # We want: min over ALL rollouts per metric independently
            all_metrics.append((chamfers[best_ch], ord_ls[best_ol], hauss[best_h]))
        if (ai + 1) % 10 == 0:
            print(f"  {ai+1} annotations, {len(all_metrics)} GT so far",
                  flush=True)

    arr = np.asarray(all_metrics)   # (n_gt, 3)
    print(f"\ntotal GT: {arr.shape[0]}")
    print()
    for name, col in [("Chamfer", 0), ("ordered_L2", 1), ("Hausdorff", 2)]:
        print(f"{name}:")
        for th in [4, 6, 8, 10, 12, 16, 20]:
            hit = int((arr[:, col] <= th).sum())
            print(f"  @{th:>2}px:  {hit}/{arr.shape[0]} = {hit/arr.shape[0]:.3f}")
        print(f"  p50/p90/p99: "
              f"{np.percentile(arr[:, col], 50):.1f}  "
              f"{np.percentile(arr[:, col], 90):.1f}  "
              f"{np.percentile(arr[:, col], 99):.1f}")
        print()

    Path(args.out).write_text(json.dumps(dict(
        ckpt=args.ckpt,
        metrics=arr.tolist(),
    ), indent=2))


if __name__ == "__main__":
    main()
