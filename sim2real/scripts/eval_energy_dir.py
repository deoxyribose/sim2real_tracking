"""Evaluate the energy-UNet + DIR pipeline against the 59 painted flagellum
labels. Reports pre-DIR sample recall and post-DIR selected recall.

Approach:
  - For each labeled real frame:
      · Load a T-frame clip centred on that frame, canonicalize to 256×256
        residual (reuses `eval_v2.coverage.canonicalize_real_frame`).
      · Run model with N noise draws → candidate pool
      · Pre-DIR recall: closest-of-any-sample Chamfer < threshold
      · Post-DIR recall: closest-of-DIR-selected Chamfer < threshold
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sim2real.dir.build_problem import BuildConfig
from sim2real.dir.run_dir import DIRRunConfig, load_model_ckpt, load_pca_scaled
from sim2real.dir.solve import SolveConfig
from sim2real.eval_v2.coverage import (
    _chamfer_polylines,
    canonicalize_real_frame,
    gt_polyline_to_canonical,
    load_real_annotations,
)
from sim2real.model.unet_energy import (
    UNetConfig, UNetEnergy, decode_curves, sample_batched_noise, unpack_pred,
)
from sim2real.dir.build_problem import Hypothesis, build_problem
from sim2real.dir.solve import solve_problem


def sample_candidates_for_frame(
    params, model, cfg_u, residual_clip, key, n_draws, score_thresh, top_k,
    pca_mean, pca_basis, temperature=1.0,
):
    """Run the model n_draws times on the residual clip, return:
        curves    (list of (K, 2)),  scores, widths, amps."""
    video = jnp.asarray(residual_clip)[None]                    # (1, T, H, W)
    all_curves, all_scores, all_widths, all_amps = [], [], [], []
    for _ in range(n_draws):
        key, k = jax.random.split(key)
        noise = sample_batched_noise(k, 1, cfg_u, temperature=temperature)
        pred = model.apply(params, video, noise, train=False)
        curves = np.asarray(decode_curves(pred, cfg_u, pca_mean, pca_basis))[0]
        f = unpack_pred(pred)
        widths = np.asarray(f["width"][0])
        amps = np.asarray(f["amp"][0])
        scores = np.asarray(jax.nn.sigmoid(f["score"][0]))
        flat_c = curves.reshape(-1, curves.shape[-2], 2)
        flat_w = widths.reshape(-1); flat_a = amps.reshape(-1); flat_s = scores.reshape(-1)
        keep = flat_s >= score_thresh
        idxs = np.where(keep)[0]
        if len(idxs) > top_k:
            idxs = idxs[np.argsort(-flat_s[idxs])[:top_k]]
        all_curves.extend([flat_c[j] for j in idxs])
        all_scores.extend([flat_s[j] for j in idxs])
        all_widths.extend([flat_w[j] for j in idxs])
        all_amps.extend([flat_a[j] for j in idxs])
    return dict(curves=all_curves, scores=np.asarray(all_scores),
                widths=np.asarray(all_widths), amps=np.asarray(all_amps))


def eval_pre_dir_recall(candidate_curves, gt_curves_canon,
                         coverage_thresh_px=8.0):
    """Pre-DIR: for each GT flagellum, is any candidate within threshold?"""
    if not gt_curves_canon:
        return dict(covered=[], recall=1.0, min_chamfer=[])
    covered = []
    min_dists = []
    for gt in gt_curves_canon:
        best = np.inf
        for pred in candidate_curves:
            d = _chamfer_polylines(pred, gt)
            if d < best:
                best = d
                if best <= coverage_thresh_px:
                    break
        min_dists.append(best)
        covered.append(best <= coverage_thresh_px)
    return dict(covered=covered, recall=float(sum(covered) / len(covered)),
                min_chamfer=min_dists)


def eval_post_dir_recall(candidates, gt_curves_canon, dir_cfg,
                          residual_clip, coverage_thresh_px=8.0):
    """Build hypothesis pool from ALL candidates as if they came from a single
    anchor frame (frame_index=0), run DIR, then compute recall AND PRECISION.

    Precision = fraction of DIR-selected skeletons within threshold of SOME GT.
    """
    T, H, W = residual_clip.shape
    hypos = []
    for i in range(len(candidates["scores"])):
        hypos.append(Hypothesis(
            frame=0,
            skeleton=candidates["curves"][i].astype(np.float32),
            width=float(candidates["widths"][i]),
            amp=float(candidates["amps"][i]),
            score=float(candidates["scores"][i]),
        ))
    if not hypos:
        return dict(n_selected=0, recall=0.0, precision=0.0)
    res = residual_clip[T // 2 : T // 2 + 1]                    # single anchor
    problem = build_problem(hypos, res, dir_cfg.build)
    sol = solve_problem(problem, dir_cfg.solve)
    sel_curves = [hypos[i].skeleton for i in sol["selected_indices"]]
    if not sel_curves:
        return dict(n_selected=0, recall=0.0, precision=0.0, selected=[])
    covered = []
    for gt in gt_curves_canon:
        best = min(_chamfer_polylines(p, gt) for p in sel_curves)
        covered.append(best <= coverage_thresh_px)
    # Precision: for each selected skeleton, is any GT within threshold?
    if not gt_curves_canon:
        precision = 0.0
    else:
        tp = 0
        for p in sel_curves:
            if min(_chamfer_polylines(p, g) for g in gt_curves_canon) <= coverage_thresh_px:
                tp += 1
        precision = tp / max(len(sel_curves), 1)
    return dict(n_selected=len(sel_curves),
                recall=float(sum(covered) / max(len(covered), 1)),
                precision=float(precision),
                selected=sel_curves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pca", default="data_cache/flagella_pca.npz")
    ap.add_argument("--n-draws", type=int, default=8)
    ap.add_argument("--score-thresh", type=float, default=0.05)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--coverage-thresh", type=float, default=8.0,
                    help="pre-DIR Chamfer threshold (canonical px)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only eval first N annotations (0 = all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dir-pick-base", type=float, default=3.0)
    ap.add_argument("--dir-score-bonus", type=float, default=30.0)
    ap.add_argument("--dir-birth", type=float, default=0.5)
    ap.add_argument("--dir-death", type=float, default=0.5)
    args = ap.parse_args()

    print(f"loading ckpt {args.ckpt}", flush=True)
    params, cfg_u = load_model_ckpt(args.ckpt)
    pca_mean, pca_basis = load_pca_scaled(args.pca)
    model = UNetEnergy(cfg=cfg_u)

    dir_cfg = DIRRunConfig(
        n_noise_draws=1,   # already exhausted at candidate gather
        build=BuildConfig(cost_mode="score_only",
                           pick_cost_base=args.dir_pick_base,
                           score_bonus=args.dir_score_bonus,
                           birth_cost=args.dir_birth,
                           death_cost=args.dir_death),
        solve=SolveConfig(time_limit_s=15.0),
    )

    print("loading real annotations...", flush=True)
    annots = load_real_annotations()
    if args.limit > 0:
        annots = annots[: args.limit]
    print(f"n annotations: {len(annots)}", flush=True)

    key = jax.random.key(42)
    pre_dir_recall_per_ann = []
    post_dir_recall_per_ann = []
    n_gt_per_ann = []
    per_ann = []

    for ai, ann in enumerate(annots):
        try:
            canon, cfg_can = canonicalize_real_frame(ann["meta"],
                                                       ann["src_width_px"],
                                                       T=cfg_u.T)
        except Exception as e:
            print(f"  [{ai:3d}] skip {ann['name']}: {e}", flush=True)
            continue
        clip = canon["clip"]                                    # (T, H_c, W_c)
        # canonical H, W may not equal cfg_u.H, W — pad if needed
        if clip.shape[1] != cfg_u.H or clip.shape[2] != cfg_u.W:
            ph = cfg_u.H - clip.shape[1]
            pw = cfg_u.W - clip.shape[2]
            if ph < 0 or pw < 0:
                print(f"  [{ai:3d}] clip too big: {clip.shape}", flush=True)
                continue
            clip = np.pad(clip, ((0, 0), (0, ph), (0, pw)), constant_values=0.0)

        # GT polylines: transform to canonical then to (y, x) skeleton (same K as our model)
        gt_curves = []
        for pl_native in ann["gt_polylines_native"]:
            gt_canon = gt_polyline_to_canonical(pl_native, ann["meta"], cfg_can,
                                                  canonical_h=cfg_u.H, canonical_w=cfg_u.W)
            if len(gt_canon) < 4:
                continue
            gt_curves.append(gt_canon)

        key, kk = jax.random.split(key)
        cand = sample_candidates_for_frame(
            params, model, cfg_u, clip, kk,
            n_draws=args.n_draws, score_thresh=args.score_thresh,
            top_k=args.top_k, pca_mean=pca_mean, pca_basis=pca_basis,
        )

        pre = eval_pre_dir_recall(cand["curves"], gt_curves,
                                    coverage_thresh_px=args.coverage_thresh)
        post = eval_post_dir_recall(cand, gt_curves, dir_cfg, clip,
                                      coverage_thresh_px=args.coverage_thresh)

        pre_dir_recall_per_ann.append(pre["recall"])
        post_dir_recall_per_ann.append(post["recall"])
        n_gt_per_ann.append(len(gt_curves))

        per_ann.append(dict(
            name=ann["name"],
            n_gt=len(gt_curves),
            n_candidates=len(cand["scores"]),
            pre_dir_recall=pre["recall"],
            post_dir_recall=post["recall"],
            post_dir_precision=post.get("precision", 0.0),
            pre_dir_min_chamfer=[float(x) for x in pre["min_chamfer"]],
            post_dir_n_selected=post["n_selected"],
        ))
        print(f"  [{ai:3d}] {ann['name']:24s}  n_gt={len(gt_curves)}  "
              f"n_cand={len(cand['scores'])}  pre_R={pre['recall']:.2f}  "
              f"post_R={post['recall']:.2f}  post_P={post.get('precision', 0):.3f}  "
              f"post_sel={post['n_selected']}",
              flush=True)

    total_pre = sum(r * n for r, n in zip(pre_dir_recall_per_ann, n_gt_per_ann)) / \
                 max(sum(n_gt_per_ann), 1)
    total_post = sum(r * n for r, n in zip(post_dir_recall_per_ann, n_gt_per_ann)) / \
                  max(sum(n_gt_per_ann), 1)
    # Precision aggregated across all selected predictions (not per-annotation)
    total_sel = sum(a.get("post_dir_n_selected", 0) for a in per_ann)
    total_tp = sum(a.get("post_dir_precision", 0) * a.get("post_dir_n_selected", 0)
                    for a in per_ann)
    total_precision = float(total_tp) / max(total_sel, 1)

    summary = dict(
        ckpt=args.ckpt,
        n_annotations=len(annots),
        n_gt_total=sum(n_gt_per_ann),
        pre_dir_recall=float(total_pre),
        post_dir_recall=float(total_post),
        post_dir_precision=total_precision,
        post_dir_n_selected_total=int(total_sel),
        coverage_thresh_px=args.coverage_thresh,
        per_annotation=per_ann,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print()
    print(f"OVERALL   pre_R = {total_pre:.3f}   post_R = {total_post:.3f}   "
          f"post_P = {total_precision:.3f}   n_sel_total = {total_sel}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
