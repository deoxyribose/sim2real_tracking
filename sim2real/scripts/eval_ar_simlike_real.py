"""Eval v16 on real annotations using sim-like preprocessing (no sigma-scale).

Compares directly against eval_ar_fast (canonicalize with sigma-scale) to
quantify the input-scale gap contribution to the sim2real recall drop.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import cv2, jax, numpy as np

from sim2real.eval_v2.coverage import (
    _chamfer_polylines, load_real_annotations,
)
from sim2real.eval_v2.simlike_preproc import simlike_canonicalize
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
    ap.add_argument("--tta-angles", nargs="+", type=float,
                    default=[0.0, -8.0, 8.0])
    ap.add_argument("--coverage-thresh", type=float, default=6.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params, cfg = load_ckpt(args.ckpt)
    backbone = UNetARBackbone(cfg=cfg)
    attach_head = AttachmentHead(cfg=cfg)
    knot_gen = KnotGenerator(cfg=cfg)
    sampler = make_sampler(cfg, backbone, attach_head, knot_gen)
    annots = load_real_annotations()
    if args.limit: annots = annots[: args.limit]

    key = jax.random.key(42)
    all_recalls, all_n_gt = [], []
    t_start = time.time()
    for ai, ann in enumerate(annots):
        try:
            clip_r, smed_r, src_h, src_w, sy, sx = simlike_canonicalize(
                ann["meta"], T=cfg.T, target_hw=(cfg.H, cfg.W))
        except Exception as e:
            continue

        # GT in target (H, W) coords — directly from cropped-native via sy, sx
        gts = []
        for pl in ann["gt_polylines_native"]:
            if len(pl) < 4: continue
            g = pl.astype(np.float32) * np.asarray([sy, sx])
            gts.append(g)
        if not gts: continue

        rollouts, key = sample_pool_one_clip(
            params, backbone, attach_head, knot_gen, cfg,
            clip_r, smed_r, list(args.tta_angles), flips=(False, True),
            n_draws=args.n_draws, n_attach=args.n_attach,
            n_rollouts=args.n_rollouts,
            score_thresh=0.02, key=key, _sampler=sampler)

        covered = []
        for g in gts:
            if not rollouts: covered.append(False); continue
            dists = [_chamfer_polylines(rl, g) for rl in rollouts]
            covered.append(min(dists) <= args.coverage_thresh)
        r = float(sum(covered) / len(covered))
        all_recalls.append(r); all_n_gt.append(len(gts))
    total = sum(r * n for r, n in zip(all_recalls, all_n_gt)) / max(sum(all_n_gt), 1)
    print(f"sim-like preprocessing (median-sub+/255, no sigma-scale, no bandpass): "
          f"pre_recall = {total:.3f}  ({sum(all_n_gt)} GT)  "
          f"{time.time()-t_start:.1f}s")
    Path(args.out).write_text(json.dumps(dict(
        ckpt=args.ckpt, preprocessing="simlike",
        pre_recall=float(total), coverage_thresh=args.coverage_thresh,
    ), indent=2))


if __name__ == "__main__":
    main()
