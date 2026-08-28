"""End-to-end verifier eval on real annotated frames.

For each frame:
  1. Canonicalize the source clip.
  2. Model forward → head outputs.
  3. Sample N flagellum + M cell candidates from each slot's Gaussians.
       (Keep only slots whose class_prob > threshold for that class.)
  4. Render each candidate on the canonical canvas.
  5. Score each candidate via reconstruction delta against the clip midpoint.
  6. Feed candidate pool + target to CP-SAT verifier.
  7. Compute precision / recall against real GT polylines.

Metric definitions:
  - GT is "found" if any SELECTED candidate has Chamfer to GT ≤ k*canonical_width.
  - A selected candidate is "correct" if its Chamfer to some GT ≤ k*canonical_width.
  - precision = correct_selected / total_selected
  - recall    = found_gt / total_gt
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sim2real.data import CANONICAL_H, CANONICAL_W, CANONICAL_TARGET_WIDTH_PX, CLASS_CELL, CLASS_FLAGELLUM
from sim2real.eval_v2.coverage import (
    canonicalize_real_frame, gt_polyline_to_canonical, load_real_annotations,
    _chamfer_polylines,
)
from sim2real.model_v2 import (
    DETRSlotModel, sample_flagellum_from_head, sample_cell_from_head,
)
from sim2real.verifier import (
    build_and_solve, VerifyConfig,
    render_flagellum_candidate, render_cell_candidate,
)


def build_candidate_pools(out: dict, n_samples: int, temperature: float,
                          key: jax.Array,
                          p_flag_thresh: float = 0.05,
                          p_cell_thresh: float = 0.05,
                          max_per_slot: int = 20):
    """From model output, sample and render up to max_per_slot candidates per slot.

    Only slots whose class prob (softmax) for that class exceeds threshold contribute.
    """
    B, S = out["class_logits"].shape[:2]
    assert B == 1
    class_probs = np.asarray(jax.nn.softmax(out["class_logits"], axis=-1))[0]  # (S, 4)
    p_flag = class_probs[:, CLASS_FLAGELLUM]
    p_cell = class_probs[:, CLASS_CELL]

    key_f, key_c = jax.random.split(key)
    f_samples = sample_flagellum_from_head(key_f, out, n_samples=n_samples, temperature=temperature)
    c_samples = sample_cell_from_head(key_c, out, n_samples=n_samples, temperature=temperature)

    f_pts = np.asarray(f_samples["pts_samples"])[0]         # (S, N, K+1, 2)
    f_widths = np.asarray(f_samples["width_samples"])[0]    # (S, N)
    f_amps = np.asarray(f_samples["amp_samples"])[0]        # (S, N)
    f_pol = np.asarray(f_samples["polarity_samples"])[0]    # (S, N) ±1
    c_centers = np.asarray(c_samples["center_samples"])[0]  # (S, N, 2)
    c_radii = np.asarray(c_samples["radius_samples"])[0]    # (S, N)
    c_amps = np.asarray(c_samples["amp_samples"])[0]        # (S, N)

    # Collect flagellum candidates from slots with p_flag > threshold
    flag_pool_pts, flag_pool_att, flag_pool_width, flag_pool_amp, flag_pool_slot = [], [], [], [], []
    for s in range(S):
        if p_flag[s] < p_flag_thresh:
            continue
        # Keep top-K by (arbitrary) — just first N samples; later we can sort by prob
        for n in range(min(n_samples, max_per_slot)):
            flag_pool_pts.append(f_pts[s, n])
            flag_pool_att.append(f_pts[s, n, 0])
            # Signed amp: positive amp is darker (canonical residual sign convention: dark=neg)
            flag_pool_width.append(max(float(f_widths[s, n]), 1.5))
            flag_pool_amp.append(float(f_amps[s, n]) * float(f_pol[s, n]))
            flag_pool_slot.append(s)

    # Cell candidates
    cell_pool_center, cell_pool_radius, cell_pool_amp = [], [], []
    for s in range(S):
        if p_cell[s] < p_cell_thresh:
            continue
        for n in range(min(n_samples, max_per_slot)):
            cell_pool_center.append(c_centers[s, n])
            cell_pool_radius.append(max(float(c_radii[s, n]), 5.0))
            cell_pool_amp.append(float(c_amps[s, n]))

    # Render candidates
    H, W = CANONICAL_H, CANONICAL_W
    if flag_pool_pts:
        flag_renders = np.stack([
            render_flagellum_candidate(np.asarray(pts), w, a, H, W)
            for pts, w, a in zip(flag_pool_pts, flag_pool_width, flag_pool_amp)
        ])
    else:
        flag_renders = np.zeros((0, H, W), dtype=np.float32)
    if cell_pool_center:
        cell_renders = np.stack([
            render_cell_candidate(np.asarray(c), r, a, H, W)
            for c, r, a in zip(cell_pool_center, cell_pool_radius, cell_pool_amp)
        ])
    else:
        cell_renders = np.zeros((0, H, W), dtype=np.float32)

    flag_dict = dict(
        renders=flag_renders,
        attachments=np.asarray(flag_pool_att, dtype=np.float32) if flag_pool_att else np.zeros((0, 2), dtype=np.float32),
        pts=np.asarray(flag_pool_pts, dtype=np.float32) if flag_pool_pts else np.zeros((0, 8, 2), dtype=np.float32),
        width=np.asarray(flag_pool_width, dtype=np.float32),
        amp=np.asarray(flag_pool_amp, dtype=np.float32),
        source_slot=np.asarray(flag_pool_slot, dtype=np.int32),
    )
    cell_dict = dict(
        renders=cell_renders,
        centers=np.asarray(cell_pool_center, dtype=np.float32) if cell_pool_center else np.zeros((0, 2), dtype=np.float32),
        radii=np.asarray(cell_pool_radius, dtype=np.float32),
    )
    return flag_dict, cell_dict, dict(p_flag=p_flag, p_cell=p_cell)


def score_selection_vs_gt(selected_pts_list: list[np.ndarray],
                           gt_polylines_canonical: list[np.ndarray],
                           thresh: float) -> dict:
    """Compute precision + recall.

    A GT is "found" if some selected has Chamfer ≤ thresh.
    A selected is "correct" if its Chamfer to some GT ≤ thresh.
    """
    if not selected_pts_list and not gt_polylines_canonical:
        return dict(precision=1.0, recall=1.0, tp=0, fp=0, fn=0)
    if not selected_pts_list:
        return dict(precision=0.0, recall=0.0, tp=0, fp=0, fn=len(gt_polylines_canonical))
    if not gt_polylines_canonical:
        return dict(precision=0.0, recall=1.0, tp=0, fp=len(selected_pts_list), fn=0)
    # Best GT for each selected + best selected for each GT
    sel_min_d = [min(_chamfer_polylines(sel, gt) for gt in gt_polylines_canonical) for sel in selected_pts_list]
    gt_min_d = [min(_chamfer_polylines(sel, gt) for sel in selected_pts_list) for gt in gt_polylines_canonical]
    tp = sum(d <= thresh for d in sel_min_d)
    fp = sum(d > thresh for d in sel_min_d)
    fn = sum(d > thresh for d in gt_min_d)
    prec = tp / max(tp + fp, 1)
    rec = sum(d <= thresh for d in gt_min_d) / max(len(gt_polylines_canonical), 1)
    return dict(precision=float(prec), recall=float(rec), tp=int(tp), fp=int(fp), fn=int(fn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--n-samples", type=int, default=20, help="Samples per slot per class")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--coverage-k", type=float, default=3.0)
    ap.add_argument("--max-cells", type=int, default=2)
    ap.add_argument("--max-flagella", type=int, default=4)
    ap.add_argument("--attach-slack", type=float, default=6.0)
    ap.add_argument("--birth-flag", type=float, default=0.5)
    ap.add_argument("--birth-cell", type=float, default=1.0)
    ap.add_argument("--p-flag-thresh", type=float, default=0.05)
    ap.add_argument("--p-cell-thresh", type=float, default=0.05)
    ap.add_argument("--solver-time", type=float, default=15.0)
    ap.add_argument("--scoring", choices=("matched", "recon", "energy"), default="energy")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = pickle.load(open(args.ckpt, "rb"))
    model = DETRSlotModel(cfg=ckpt["model_cfg"])
    print(f"[ckpt] step={ckpt['step']}, cfg={ckpt['model_cfg']}")

    ann = load_real_annotations()
    print(f"[data] {len(ann)} real annotated frames")

    @jax.jit
    def fwd(p, c, e, k):
        return model.apply(p, c, e, rngs={"slots": k})

    cfg = VerifyConfig(
        max_cells=args.max_cells, max_flagella=args.max_flagella,
        attach_slack_px=args.attach_slack,
        birth_prior_per_flag=args.birth_flag, birth_prior_per_cell=args.birth_cell,
        solver_time_limit_s=args.solver_time,
        scoring=args.scoring,
    )

    key = jax.random.PRNGKey(0)
    results = []
    total_tp = total_fp = total_fn = 0
    thresh = args.coverage_k * CANONICAL_TARGET_WIDTH_PX

    t0 = time.time()
    for i, entry in enumerate(ann):
        try:
            canon, cfg_c = canonicalize_real_frame(entry["meta"], entry["src_width_px"], T=args.T)
        except Exception as e:
            print(f"  skip {entry['name']}: {e}")
            continue
        clip = jnp.asarray(canon["clip"])[None]
        energy = jnp.asarray(canon["energy"])[None]
        key, sub = jax.random.split(key)
        out = fwd(ckpt["params"], clip, energy, sub)
        key, sub2 = jax.random.split(key)
        flag_dict, cell_dict, probs = build_candidate_pools(
            out, n_samples=args.n_samples, temperature=args.temperature, key=sub2,
            p_flag_thresh=args.p_flag_thresh, p_cell_thresh=args.p_cell_thresh,
        )
        # Target = "beat envelope": most-negative across time (flagellum darkens pixels
        # transiently as it passes through). Cells contribute a static negative floor.
        # This is more informative than a single-frame midpoint because a candidate curve
        # SOMEWHERE in its beat envelope should overlap the envelope of the real data.
        target = canon["clip"].min(axis=0)

        res = build_and_solve(flag_dict, cell_dict, target, cfg=cfg, energy_map=canon["energy"])

        selected_pts = [flag_dict["pts"][k] for k in res["selected_flag_idx"]]
        gt_canonical = [gt_polyline_to_canonical(p, entry["meta"], cfg_c) for p in entry["gt_polylines_native"]]
        pr_rec = score_selection_vs_gt(selected_pts, gt_canonical, thresh)

        results.append(dict(
            name=entry["name"], sequence=entry["meta"]["sequence"],
            n_flag_cand=int(flag_dict["renders"].shape[0]),
            n_cell_cand=int(cell_dict["renders"].shape[0]),
            n_selected_flag=len(res["selected_flag_idx"]),
            n_selected_cell=len(res["selected_cell_idx"]),
            solver_status=res["status"],
            solver_time_s=res["wall_time_s"],
            objective=res["objective"],
            n_gt=len(gt_canonical),
            **pr_rec,
        ))
        total_tp += pr_rec["tp"]; total_fp += pr_rec["fp"]; total_fn += pr_rec["fn"]
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(ann) - i - 1)
            print(f"  {i+1}/{len(ann)}: "
                  f"cand_f={flag_dict['renders'].shape[0]}, cand_c={cell_dict['renders'].shape[0]}, "
                  f"sel_f={len(res['selected_flag_idx'])}, sel_c={len(res['selected_cell_idx'])}  "
                  f"prec={pr_rec['precision']:.2f} rec={pr_rec['recall']:.2f}  eta={eta:.0f}s")

    prec = total_tp / max(total_tp + total_fp, 1)
    rec = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)
    print(f"\n[overall] tp={total_tp} fp={total_fp} fn={total_fn}  precision={prec:.3f} recall={rec:.3f} F1={f1:.3f}")

    # Per-sequence
    by_seq: dict = {}
    for r in results:
        s = by_seq.setdefault(r["sequence"], dict(tp=0, fp=0, fn=0))
        s["tp"] += r["tp"]; s["fp"] += r["fp"]; s["fn"] += r["fn"]
    print("\n[per-sequence]")
    for seq, s in sorted(by_seq.items()):
        p = s["tp"] / max(s["tp"] + s["fp"], 1)
        r_ = s["tp"] / max(s["tp"] + s["fn"], 1)
        print(f"  {seq[-45:]:<45s}  tp={s['tp']:>2} fp={s['fp']:>2} fn={s['fn']:>2}  P={p:.2f} R={r_:.2f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "overall": dict(precision=prec, recall=rec, f1=f1, tp=total_tp, fp=total_fp, fn=total_fn),
                "coverage_k": args.coverage_k,
                "per_frame": results,
                "per_sequence": {k: {**v, "precision": v["tp"] / max(v["tp"]+v["fp"], 1),
                                     "recall": v["tp"] / max(v["tp"]+v["fn"], 1)} for k, v in by_seq.items()},
            }, f, indent=2)
        print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
