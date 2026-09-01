"""Eval with test-time augmentation: horizontal flip + small rotations
applied to real input; rollouts inverse-transformed back to canonical
coordinates; pooled candidate pool.

Same base logic as eval_ar, but each annotation runs `n_tta` extra
augmented versions of the input alongside the original.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from sim2real.data.canonicalize import CANONICAL_H, CANONICAL_W
from sim2real.eval_v2.coverage import (
    _chamfer_polylines, canonicalize_real_frame, gt_polyline_to_canonical,
    load_real_annotations,
)
from sim2real.model.unet_ar import (
    AttachmentHead, KnotGenerator, UNetARBackbone,
    sample_batched_noise, unpack_attachment,
)
from sim2real.scripts.eval_ar import load_ckpt, rollout_batch, sample_attachments


def rotate_clip(clip: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate each frame by angle_deg (positive=CCW), keep same shape."""
    T, H, W = clip.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle_deg, 1.0)
    return np.stack([cv2.warpAffine(clip[t], M, (W, H)) for t in range(T)], axis=0)


def flip_clip(clip: np.ndarray) -> np.ndarray:
    return clip[:, :, ::-1].copy()


def inverse_transform_rollout(rollout: np.ndarray, H: int, W: int,
                               angle_deg: float, flipped: bool) -> np.ndarray:
    """Rollout is in the AUGMENTED image coords. Un-augment back to canonical."""
    x = rollout.copy()
    if flipped:
        x[:, 1] = (W - 1) - x[:, 1]
    if angle_deg != 0:
        # Undo rotation: rotate rollout by -angle around center
        M = cv2.getRotationMatrix2D((W / 2, H / 2), -angle_deg, 1.0)
        pts = np.stack([x[:, 1], x[:, 0], np.ones(x.shape[0])], axis=-1)  # (K, 3)
        xy = pts @ M.T                                                     # (K, 2)
        x = np.stack([xy[:, 1], xy[:, 0]], axis=-1)
    return x


def run_augmented(params, backbone, attach_head, knot_gen, cfg,
                    clip: np.ndarray, smed: np.ndarray, key,
                    n_draws: int, n_attach: int, n_rollouts: int,
                    score_thresh: float, angle_deg: float, flipped: bool,
                    return_scores: bool = False):
    """Run one augmented version of an image; return rollouts in ORIGINAL
    (canonical / model-canvas) coordinates. If return_scores, also return
    a parallel list of per-rollout attachment scores (inherited from the
    seed grid cell — all rollouts from one attachment share its score)."""
    aug_clip = flip_clip(clip) if flipped else clip
    if angle_deg != 0:
        aug_clip = rotate_clip(aug_clip, angle_deg)
    # Same augmentation on static median
    aug_smed = flip_clip(smed[None])[0] if flipped else smed
    if angle_deg != 0:
        aug_smed = rotate_clip(aug_smed[None], angle_deg)[0]

    video = jnp.asarray(aug_clip)[None]
    smed_in = jnp.asarray(aug_smed)[None, ..., None]
    rollouts_orig = []
    scores_orig = []
    for _ in range(n_draws):
        key, k = jax.random.split(key)
        noise = sample_batched_noise(k, 1, cfg)
        full_res, grid = backbone.apply(params["backbone"], video, noise,
                                          smed_in, train=False)
        attach = attach_head.apply(params["attach"], grid)
        f = unpack_attachment(attach[0])
        atts, att_scores = sample_attachments(f, n_attach, cfg)
        keep = att_scores >= score_thresh
        atts = atts[keep]; att_scores = att_scores[keep]
        if len(atts) == 0: continue
        for _ in range(n_rollouts):
            key, kr = jax.random.split(key)
            rollout_keys = jax.random.split(kr, len(atts))
            rollouts = rollout_batch(full_res[0], params["knot"], knot_gen,
                                      cfg, jnp.asarray(atts), rollout_keys, 1.0)
            for r, s in zip(rollouts, att_scores):
                inv = inverse_transform_rollout(np.asarray(r), cfg.H, cfg.W,
                                                  angle_deg, flipped)
                rollouts_orig.append(inv)
                scores_orig.append(float(s))
    if return_scores:
        return rollouts_orig, scores_orig, key
    return rollouts_orig, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-draws", type=int, default=4)
    ap.add_argument("--n-attach", type=int, default=8)
    ap.add_argument("--n-rollouts", type=int, default=4)
    ap.add_argument("--score-thresh", type=float, default=0.02)
    ap.add_argument("--coverage-thresh", type=float, default=6.0)
    ap.add_argument("--tta-angles", nargs="+", type=float,
                    default=[0.0, -10.0, 10.0])
    ap.add_argument("--tta-flip", action="store_true", default=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params, cfg = load_ckpt(args.ckpt)
    backbone = UNetARBackbone(cfg=cfg)
    attach_head = AttachmentHead(cfg=cfg)
    knot_gen = KnotGenerator(cfg=cfg)
    annots = load_real_annotations()
    print(f"n annotations: {len(annots)}", flush=True)

    key = jax.random.key(42)
    all_pre_recalls, all_n_gt = [], []
    per_ann = []
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

        gt_curves = []
        for pl in ann["gt_polylines_native"]:
            gc = gt_polyline_to_canonical(pl, ann["meta"], cfg_can,
                                            canonical_h=CANONICAL_H,
                                            canonical_w=CANONICAL_W)
            if len(gc) >= 4:
                gt_curves.append(gc * np.asarray([scale_y, scale_x]))
        if not gt_curves:
            continue

        all_rollouts = []
        for angle in args.tta_angles:
            for flipped in ([False, True] if args.tta_flip else [False]):
                rollouts, key = run_augmented(
                    params, backbone, attach_head, knot_gen, cfg,
                    clip_r, smed_r, key,
                    args.n_draws, args.n_attach, args.n_rollouts,
                    args.score_thresh, angle, flipped)
                all_rollouts.extend(rollouts)

        covered = []
        min_ds = []
        for gt in gt_curves:
            dists = [_chamfer_polylines(r, gt) for r in all_rollouts]
            m = min(dists) if dists else float("inf")
            covered.append(m <= args.coverage_thresh)
            min_ds.append(m)
        recall = float(sum(covered) / len(covered))
        all_pre_recalls.append(recall); all_n_gt.append(len(gt_curves))
        per_ann.append(dict(name=ann["name"], n_gt=len(gt_curves),
                              n_rollouts=len(all_rollouts),
                              pre_recall=recall,
                              min_chamfers=[float(x) for x in min_ds]))
        print(f"  [{ai:3d}] {ann['name']:20s}  n_gt={len(gt_curves)}  "
              f"n_roll={len(all_rollouts)}  recall={recall:.2f}  "
              f"min_d={[round(x,1) for x in min_ds]}",
              flush=True)

    total = sum(r * n for r, n in zip(all_pre_recalls, all_n_gt)) / \
             max(sum(all_n_gt), 1)
    print(f"\nOVERALL pre_recall_tta = {total:.3f} @ {args.coverage_thresh}px")
    Path(args.out).write_text(json.dumps(dict(
        ckpt=args.ckpt, pre_recall=float(total),
        coverage_thresh=args.coverage_thresh,
        tta_angles=args.tta_angles, tta_flip=bool(args.tta_flip),
        per_annotation=per_ann,
    ), indent=2))


if __name__ == "__main__":
    main()
