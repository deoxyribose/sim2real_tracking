"""Validate the tracking IP on REAL data using the 59 painted labels.

The labels are single-frame. We use each labeled real frame's GT flagella
as "static" across a T=16 window (reasonable approx for the short flagellum
sequences), add per-frame noise, plus distractors, and check that the IP
recovers ~n_labeled tracks.

For each labeled frame:
  1. Load the video sequence around it (T=16 frames of raw image).
  2. Take the labeled GT polylines at the anchor frame.
  3. At every frame in [0, T): produce n_noisy copies of each GT polyline
     (noise + small per-frame drift), plus n_distractors random polylines.
  4. Build ILP + solve → measure real_tracks / n_gt recovery.
"""
from __future__ import annotations
import argparse, cv2
from pathlib import Path

import numpy as np

from sim2real.dir.build_problem import BuildConfig, Hypothesis, build_problem
from sim2real.dir.solve import SolveConfig, solve_problem
from sim2real.data.canonicalize import CANONICAL_H, CANONICAL_W
from sim2real.eval_v2.coverage import (
    canonicalize_real_frame, gt_polyline_to_canonical, load_real_annotations,
)


def noisy_copy_with_drift(curve: np.ndarray, t_norm: float,
                            noise_sigma: float, drift_px: float,
                            rng: np.random.Generator) -> np.ndarray:
    """Perturb per-knot Gaussian(σ). Also apply a small linear drift over t."""
    dy = drift_px * (t_norm - 0.5)
    dx = drift_px * 0.3 * (t_norm - 0.5)
    return curve + np.array([dy, dx]) + rng.normal(0, noise_sigma, curve.shape)


def random_distractor(H, W, K, rng):
    y0, x0 = rng.uniform(20, H - 20), rng.uniform(20, W - 20)
    angle = rng.uniform(0, 2 * np.pi)
    knots = [(y0, x0)]
    for _ in range(K):
        angle += rng.normal(0, 0.4)
        step = rng.uniform(2, 5)
        y0 += step * np.sin(angle); x0 += step * np.cos(angle)
        knots.append((y0, x0))
    return np.array(knots, dtype=np.float32)


def run_one_annotation(ann: dict, T_video: int, n_noisy: int,
                         noise_sigma: float, drift_px: float,
                         n_distractors: int, model_H: int, model_W: int,
                         build_cfg, solve_cfg, rng: np.random.Generator,
                         flag_width: float = 1.5, flag_amp: float = -0.15):
    """Return dict with n_gt, n_tracks, n_real_tracks, purities."""
    canon, cfg_can = canonicalize_real_frame(ann["meta"], ann["src_width_px"],
                                                T=T_video)
    clip = canon["clip"]                                          # (T, H, W)
    src_h, src_w = clip.shape[1], clip.shape[2]
    sy, sx = model_H / src_h, model_W / src_w
    clip_r = np.stack([cv2.resize(clip[t], (model_W, model_H),
                                    interpolation=cv2.INTER_AREA)
                        for t in range(clip.shape[0])], axis=0).astype(np.float32)

    gts = []
    for pl in ann["gt_polylines_native"]:
        g = gt_polyline_to_canonical(pl, ann["meta"], cfg_can,
                                       canonical_h=CANONICAL_H,
                                       canonical_w=CANONICAL_W)
        if len(g) >= 4:
            # Resample to K=25 knots to match sim curve length (encodes as polyline)
            n_out = 25
            seg = np.linalg.norm(np.diff(g, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            if cum[-1] < 1e-6: continue
            tgt = np.linspace(0, cum[-1], n_out)
            ys = np.interp(tgt, cum, g[:, 0]); xs = np.interp(tgt, cum, g[:, 1])
            g_r = np.stack([ys, xs], axis=-1) * np.array([sy, sx])
            gts.append(g_r.astype(np.float32))
    n_gt = len(gts)
    if n_gt == 0:
        return None

    all_hypos, hypo_gt_flag = [], []
    for t in range(T_video):
        t_norm = t / max(T_video - 1, 1)
        for g in gts:
            for _ in range(n_noisy):
                nc = noisy_copy_with_drift(g, t_norm, noise_sigma, drift_px, rng)
                all_hypos.append(Hypothesis(
                    frame=t, skeleton=nc.astype(np.float32),
                    width=flag_width, amp=flag_amp,
                    score=float(0.9 - rng.uniform(0, 0.1))))
                hypo_gt_flag.append(True)
        for _ in range(n_distractors):
            d = random_distractor(model_H, model_W, 24, rng)
            all_hypos.append(Hypothesis(
                frame=t, skeleton=d.astype(np.float32),
                width=1.5, amp=-1.0,
                score=float(0.3 + rng.uniform(0, 0.3))))
            hypo_gt_flag.append(False)

    residuals = np.zeros((T_video, model_H, model_W), dtype=np.float32)
    for t in range(T_video):
        residuals[t] = clip_r[t] - np.median(clip_r, axis=0)

    problem = build_problem(all_hypos, residuals, build_cfg)
    sol = solve_problem(problem, solve_cfg)
    purities = []
    for tr in sol["tracks"]:
        p = sum(1 for i in tr if hypo_gt_flag[i]) / max(len(tr), 1)
        purities.append(p)
    sel_gt = sum(1 for i in sol["selected_indices"] if hypo_gt_flag[i])
    sel_di = sum(1 for i in sol["selected_indices"] if not hypo_gt_flag[i])
    n_real_tracks = sum(1 for p in purities if p >= 0.5)
    return dict(n_gt=n_gt, n_tracks=len(sol["tracks"]),
                  n_real_tracks=n_real_tracks, sel_gt=sel_gt, sel_di=sel_di,
                  purities=purities, name=ann["name"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T-video", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = all annotations")
    ap.add_argument("--n-noisy-per-gt", type=int, default=6)
    ap.add_argument("--noise-sigma", type=float, default=1.5)
    ap.add_argument("--drift-px", type=float, default=2.0)
    ap.add_argument("--n-distractors-per-frame", type=int, default=20)
    ap.add_argument("--cost-mode", choices=["score_only", "recon+score"],
                    default="recon+score")
    ap.add_argument("--birth-cost", type=float, default=200.0)
    ap.add_argument("--death-cost", type=float, default=200.0)
    ap.add_argument("--score-bonus", type=float, default=100.0)
    ap.add_argument("--pick-cost-base", type=float, default=5.0)
    ap.add_argument("--overlap-frac", type=float, default=0.3)
    ap.add_argument("--link-max-dist", type=float, default=18.0)
    ap.add_argument("--link-max-gap", type=int, default=2)
    ap.add_argument("--link-cost-scale", type=float, default=0.15)
    ap.add_argument("--flag-width", type=float, default=1.5)
    ap.add_argument("--flag-amp", type=float, default=-0.15,
                    help="signed residual amp at flagellum; real is DARKER than BG → negative")
    ap.add_argument("--model-h", type=int, default=128,
                    help="target image size (matches AR model H)")
    args = ap.parse_args()

    annots = load_real_annotations()
    if args.limit: annots = annots[: args.limit]
    print(f"n annotations: {len(annots)}")

    build_cfg = BuildConfig(
        cost_mode=args.cost_mode,
        pick_cost_base=args.pick_cost_base,
        score_bonus=args.score_bonus,
        max_pair_overlap_frac=args.overlap_frac,
        birth_cost=args.birth_cost, death_cost=args.death_cost,
        link_max_gap=args.link_max_gap, link_max_dist=args.link_max_dist,
        link_cost_scale=args.link_cost_scale, link_gap_cost_factor=1.5,
    )
    solve_cfg = SolveConfig(time_limit_s=15.0, num_workers=8)

    rng = np.random.default_rng(0)
    results = []
    for ai, ann in enumerate(annots):
        try:
            r = run_one_annotation(ann, args.T_video, args.n_noisy_per_gt,
                                     args.noise_sigma, args.drift_px,
                                     args.n_distractors_per_frame,
                                     args.model_h, args.model_h,
                                     build_cfg, solve_cfg, rng,
                                     args.flag_width, args.flag_amp)
        except Exception as e:
            print(f"  [{ai:3d}] {ann['name']} — skipped ({e})")
            continue
        if r is None:
            continue
        results.append(r)
        print(f"  [{ai:3d}] {r['name']}: n_gt={r['n_gt']} tracks={r['n_tracks']}"
              f" real={r['n_real_tracks']} sel {r['sel_gt']}GT+{r['sel_di']}dist"
              f" purities={[round(p,2) for p in r['purities']]}",
              flush=True)
    if not results:
        print("no results"); return
    n_gt_avg = np.mean([r["n_gt"] for r in results])
    n_tracks_avg = np.mean([r["n_tracks"] for r in results])
    n_real_avg = np.mean([r["n_real_tracks"] for r in results])
    perfect = sum(1 for r in results if r["n_real_tracks"] == r["n_gt"])
    print(f"\n{len(results)} valid annotations")
    print(f"  avg n_gt={n_gt_avg:.2f}  avg n_tracks={n_tracks_avg:.2f}"
          f"  avg real_tracks={n_real_avg:.2f}"
          f"  perfect={perfect}/{len(results)} = {perfect/len(results):.2%}")


if __name__ == "__main__":
    main()
