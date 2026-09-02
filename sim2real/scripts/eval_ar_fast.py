"""Fast recall eval using the JIT'd batched sampler (ar_batched)."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import cv2, jax, numpy as np

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
    ap.add_argument("--coverage-thresh", type=float, default=6.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params, cfg = load_ckpt(args.ckpt)
    backbone = UNetARBackbone(cfg=cfg)
    attach_head = AttachmentHead(cfg=cfg)
    knot_gen = KnotGenerator(cfg=cfg)
    sampler = make_sampler(cfg, backbone, attach_head, knot_gen,
                             temperature=args.temperature)
    annots = load_real_annotations()
    if args.limit: annots = annots[: args.limit]

    key = jax.random.key(42)
    all_recalls, all_n_gt = [], []
    per_ann = []
    t_total = time.time()
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
            score_thresh=args.score_thresh, key=key, _sampler=sampler)

        covered, mins = [], []
        for g in gts:
            if not rollouts:
                covered.append(False); mins.append(float("inf")); continue
            dists = [_chamfer_polylines(rl, g) for rl in rollouts]
            m = min(dists)
            covered.append(m <= args.coverage_thresh); mins.append(m)
        r = float(sum(covered) / len(covered))
        all_recalls.append(r); all_n_gt.append(len(gts))
        per_ann.append(dict(name=ann["name"], n_gt=len(gts),
                              n_rollouts=len(rollouts),
                              pre_recall=r,
                              min_chamfers=[float(x) for x in mins]))
    total = sum(r * n for r, n in zip(all_recalls, all_n_gt)) / max(sum(all_n_gt), 1)
    dt = time.time() - t_total
    print(f"pre_recall = {total:.3f}  ({sum(all_n_gt)} GT)  {dt:.1f}s")
    Path(args.out).write_text(json.dumps(dict(
        ckpt=args.ckpt, pre_recall=float(total),
        coverage_thresh=args.coverage_thresh,
        tta_angles=args.tta_angles, temperature=args.temperature,
        n_annotations=len(annots), per_annotation=per_ann,
    ), indent=2))


if __name__ == "__main__":
    main()
