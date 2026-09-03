"""Measure post-DIR recall on the AR pipeline.

Pre-DIR recall = 1.000 (any candidate in pool within Chamfer). Post-DIR asks:
does CP-SAT actually SELECT a good candidate for each GT flagellum? The
pool contains many overlapping rollouts of varying quality; the ILP picks
a non-overlapping subset.

For each real annotation:
  1. Run AR + wide TTA (as champion recipe) to build a rollout pool.
  2. Wrap each rollout as a Hypothesis (frame=0, uniform width/amp).
  3. Build ILP with same-frame at-most-one, no temporal links (all frame 0).
  4. Solve; take selected subset.
  5. Recall = fraction of GT covered by SELECTED subset (not full pool).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import jax
import numpy as np

from sim2real.data.canonicalize import CANONICAL_H, CANONICAL_W
from sim2real.dir.build_problem import BuildConfig, Hypothesis, build_problem
from sim2real.dir.solve import SolveConfig, solve_problem
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
    ap.add_argument("--coverage-thresh", type=float, default=6.0)
    ap.add_argument("--cost-mode", choices=["score_only", "recon+score"],
                    default="score_only")
    ap.add_argument("--pool-strategy", choices=["uniform", "top_score"],
                    default="uniform")
    ap.add_argument("--tta-angles", nargs="+", type=float,
                    default=[0.0, -5.0, 5.0, -10.0, 10.0, -15.0, 15.0])
    ap.add_argument("--width-px", type=float, default=1.5)
    ap.add_argument("--amp", type=float, default=-1.0,
                    help="signed amp: real flagella are darker than BG → negative")
    ap.add_argument("--pick-cost-base", type=float, default=1.0)
    ap.add_argument("--score-bonus", type=float, default=100.0)
    ap.add_argument("--overlap-frac", type=float, default=0.4)
    ap.add_argument("--birth-cost", type=float, default=0.0,
                    help="zero for single-frame pool (no temporal links)")
    ap.add_argument("--death-cost", type=float, default=0.0)
    ap.add_argument("--pool-cap", type=int, default=500,
                    help="cap on rollouts per clip fed to ILP; uniform-subsample if exceeded")
    ap.add_argument("--preproc", choices=["canonical", "simlike"],
                    default="canonical")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params, cfg = load_ckpt(args.ckpt)
    backbone = UNetARBackbone(cfg=cfg)
    attach_head = AttachmentHead(cfg=cfg)
    knot_gen = KnotGenerator(cfg=cfg)
    annots = load_real_annotations()
    print(f"n annotations: {len(annots)}", flush=True)

    build_cfg = BuildConfig(
        cost_mode=args.cost_mode,
        pick_cost_base=args.pick_cost_base,
        score_bonus=args.score_bonus,
        max_pair_overlap_frac=args.overlap_frac,
        birth_cost=args.birth_cost,
        death_cost=args.death_cost,
    )
    solve_cfg = SolveConfig(time_limit_s=15.0, num_workers=8)

    key = jax.random.key(42)
    all_pre_recalls, all_post_recalls, all_n_gt = [], [], []
    per_ann = []
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
        else:  # simlike
            try:
                clip_r, smed_r, src_h, src_w, sy, sx = simlike_canonicalize(
                    ann["meta"], T=cfg.T, target_hw=(cfg.H, cfg.W))
            except Exception:
                continue
            gt_curves = []
            for pl in ann["gt_polylines_native"]:
                if len(pl) >= 4:
                    gt_curves.append(pl.astype(np.float32) * np.asarray([sy, sx]))
        if not gt_curves:
            continue

        # Build rollout pool with wide TTA, tracking per-rollout attachment scores.
        pool, pool_scores = [], []
        for angle in args.tta_angles:
            for flipped in (False, True):
                r, s, key = run_augmented(params, backbone, attach_head, knot_gen, cfg,
                                            clip_r, smed_r, key,
                                            args.n_draws, args.n_attach,
                                            args.n_rollouts, args.score_thresh,
                                            angle, flipped, return_scores=True)
                pool.extend(r); pool_scores.extend(s)
        if not pool:
            continue

        # Cap pool size — DIR overlap constraints scale O(N²).
        if len(pool) > args.pool_cap:
            if args.pool_strategy == "top_score":
                order = np.argsort(-np.asarray(pool_scores))[: args.pool_cap]
            else:  # uniform
                order = np.linspace(0, len(pool) - 1, args.pool_cap).astype(int)
            pool = [pool[i] for i in order]
            pool_scores = [pool_scores[i] for i in order]

        # Pre-DIR recall (candidate pool)
        pre_covered, min_ds = [], []
        for gt in gt_curves:
            dists = [_chamfer_polylines(rl, gt) for rl in pool]
            m = min(dists)
            pre_covered.append(m <= args.coverage_thresh); min_ds.append(m)
        pre_recall = float(sum(pre_covered) / len(pre_covered))

        # Wrap as Hypotheses with per-rollout attachment scores.
        hypos = [Hypothesis(frame=0, skeleton=rl.astype(np.float32),
                             width=args.width_px, amp=args.amp, score=float(sc))
                  for rl, sc in zip(pool, pool_scores)]
        residual = clip_r[cfg.T // 2] - smed_r   # (H, W) signed residual
        residuals = residual[None]              # (1, H, W)
        try:
            problem = build_problem(hypos, residuals, build_cfg)
            solution = solve_problem(problem, solve_cfg)
        except Exception as e:
            print(f"  [{ai:3d}] DIR failed: {e}", flush=True)
            continue

        selected = solution["selected_indices"]
        sel_rollouts = [pool[i] for i in selected]

        post_covered, post_min_ds = [], []
        for gt in gt_curves:
            if not sel_rollouts:
                post_covered.append(False); post_min_ds.append(float("inf"))
                continue
            dists = [_chamfer_polylines(rl, gt) for rl in sel_rollouts]
            m = min(dists)
            post_covered.append(m <= args.coverage_thresh); post_min_ds.append(m)
        post_recall = float(sum(post_covered) / len(post_covered))

        all_pre_recalls.append(pre_recall); all_post_recalls.append(post_recall)
        all_n_gt.append(len(gt_curves))
        per_ann.append(dict(
            name=ann["name"], n_gt=len(gt_curves), n_pool=len(pool),
            n_selected=len(sel_rollouts),
            pre_recall=pre_recall, post_recall=post_recall,
            pre_min=[float(x) for x in min_ds],
            post_min=[float(x) for x in post_min_ds],
        ))
        print(f"  [{ai:3d}] {ann['name']:20s}  n_gt={len(gt_curves)}  "
              f"pool={len(pool)}  sel={len(sel_rollouts)}  "
              f"pre={pre_recall:.2f}  post={post_recall:.2f}",
              flush=True)

    pre_total = sum(r * n for r, n in zip(all_pre_recalls, all_n_gt)) / \
                 max(sum(all_n_gt), 1)
    post_total = sum(r * n for r, n in zip(all_post_recalls, all_n_gt)) / \
                  max(sum(all_n_gt), 1)
    print(f"\nOVERALL pre_recall  = {pre_total:.3f}  ({sum(all_n_gt)} GT)")
    print(f"OVERALL post_recall = {post_total:.3f}  @ {args.coverage_thresh}px")
    Path(args.out).write_text(json.dumps(dict(
        ckpt=args.ckpt, pre_recall=float(pre_total),
        post_recall=float(post_total),
        coverage_thresh=args.coverage_thresh,
        tta_angles=args.tta_angles,
        build_cfg=dict(pick_cost_base=args.pick_cost_base,
                        score_bonus=args.score_bonus,
                        overlap_frac=args.overlap_frac),
        per_annotation=per_ann,
    ), indent=2))


if __name__ == "__main__":
    main()
