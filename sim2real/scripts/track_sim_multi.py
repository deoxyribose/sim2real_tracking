"""Run tracking on many sim seeds side-by-side. Selected splines drawn ON
TOP of GT (with outline) so they're clearly visible."""
from __future__ import annotations
import argparse
from pathlib import Path

import jax, numpy as np, matplotlib.pyplot as plt

from sim2real.dir.build_problem import BuildConfig, Hypothesis, build_problem
from sim2real.dir.solve import SolveConfig, solve_problem
from sim2real.model.unet_ar import (
    AttachmentHead, KnotGenerator, UNetARBackbone,
)
from sim2real.scripts.ar_batched import make_sampler, sample_pool_one_clip
from sim2real.scripts.eval_ar import load_ckpt
from sim2real.scripts.track_sim_video import make_angle_noisy_candidates
from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip


def track_one(seed: int, T_video: int, cfg: DiverseSimConfig, build_cfg,
                solve_cfg, gt_noise_n: int, angle_sigma: float,
                step_sigma: float, rng: np.random.Generator,
                model=None):
    """model = None → noisy-GT candidates; otherwise dict with
    (params, ar_cfg, backbone, attach_head, knot_gen, sampler)."""
    out = sample_clip(jax.random.key(seed), cfg)
    clip_raw = np.asarray(out["clip_raw"]).astype(np.float32)
    gt_curves = np.asarray(out["curves"])
    alive = np.asarray(out["flagella"]["alive"])
    gt_widths = np.asarray(out["flagella"]["width"])
    gt_amps = np.asarray(out["flagella"]["amp"])
    n_alive = int(alive.sum())
    if n_alive == 0:
        return None
    # Median amp/width for model rollout hypotheses (model doesn't predict them)
    roll_width = float(np.median(gt_widths[alive]))
    roll_amp = float(np.median(gt_amps[alive]))

    anchor_frames = list(range(2, T_video - 2))
    all_hypos: list[Hypothesis] = []
    per_frame_rollouts: dict[int, list[np.ndarray]] = {}
    if model is None:
        # noisy-GT candidates
        for k in anchor_frames:
            per_frame_rollouts[k] = []
            for j in range(gt_curves.shape[1]):
                if not bool(alive[j]): continue
                g = gt_curves[k, j]
                cands = make_angle_noisy_candidates(g, gt_noise_n, angle_sigma,
                                                      step_sigma, rng)
                for c in cands:
                    per_frame_rollouts[k].append(c)
                    all_hypos.append(Hypothesis(
                        frame=k, skeleton=c.astype(np.float32),
                        width=float(gt_widths[j]), amp=float(gt_amps[j]),
                        score=0.9))
    else:
        params = model["params"]; ar_cfg = model["ar_cfg"]
        backbone = model["backbone"]; attach_head = model["attach_head"]
        knot_gen = model["knot_gen"]; sampler = model["sampler"]
        T_win = ar_cfg.T
        key = jax.random.key(seed)
        for k in anchor_frames:
            i0 = k - T_win // 2; i1 = i0 + T_win
            win_raw = clip_raw[i0:i1]
            med = np.median(win_raw, axis=0)
            win_median = win_raw - med
            smed = med.astype(np.float32)
            rollouts, scores, key = sample_pool_one_clip(
                params, backbone, attach_head, knot_gen, ar_cfg,
                win_median.astype(np.float32), smed,
                [0.0, -8.0, 8.0], flips=(False, True),
                n_draws=2, n_attach=8, n_rollouts=4,
                score_thresh=0.02, key=key, _sampler=sampler,
                return_scores=True)
            pool_cap = 30
            if len(rollouts) > pool_cap:
                order = np.argsort(-np.asarray(scores))[:pool_cap]
                rollouts = [rollouts[i] for i in order]
                scores = [scores[i] for i in order]
            per_frame_rollouts[k] = rollouts
            for rl, sc in zip(rollouts, scores):
                all_hypos.append(Hypothesis(
                    frame=k, skeleton=rl.astype(np.float32),
                    width=roll_width, amp=roll_amp, score=float(sc)))
    residuals = np.zeros((len(anchor_frames), cfg.H, cfg.W), dtype=np.float32)
    for i, k in enumerate(anchor_frames):
        residuals[i] = clip_raw[k] - np.median(clip_raw, axis=0)
    for h in all_hypos:
        h.frame = anchor_frames.index(h.frame)
    problem = build_problem(all_hypos, residuals, build_cfg)
    sol = solve_problem(problem, solve_cfg)

    hypo_track = [-1] * len(all_hypos)
    for tid, tr in enumerate(sol["tracks"]):
        for hi in tr: hypo_track[hi] = tid

    return dict(seed=seed, n_alive=n_alive,
                  clip_raw=clip_raw, gt_curves=gt_curves, alive=alive,
                  anchor_frames=anchor_frames, all_hypos=all_hypos,
                  per_frame_rollouts=per_frame_rollouts,
                  sol=sol, hypo_track=hypo_track,
                  n_tracks=len(sol["tracks"]))


def render_row(fig, gs, row, r, mid_frames, track_colors, alive, gt_curves,
                 all_hypos, sol, hypo_track, clip_raw):
    for i, k in enumerate(mid_frames):
        ax = fig.add_subplot(gs[row, i])
        img = clip_raw[k]
        lo, hi = np.percentile(img, [1, 99])
        gray = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        ax.imshow(gray, cmap="gray")
        # GT (thin green line, less prominent than selected)
        for j in range(gt_curves.shape[1]):
            if not bool(alive[j]): continue
            g = gt_curves[k, j]
            ax.plot(g[:, 1], g[:, 0], "-", color="#33ff44", linewidth=1.2,
                    alpha=0.7)
        # Selected DIR splines drawn ON TOP with dark outline for pop
        ft = r["anchor_frames"].index(k) if k in r["anchor_frames"] else -1
        if ft >= 0:
            for hi in sol["selected_indices"]:
                if all_hypos[hi].frame != ft: continue
                col = track_colors[hypo_track[hi]]
                # Outline (black) then colored line on top
                ax.plot(all_hypos[hi].skeleton[:, 1],
                        all_hypos[hi].skeleton[:, 0],
                        "-", color="black", linewidth=3.6, alpha=0.9)
                ax.plot(all_hypos[hi].skeleton[:, 1],
                        all_hypos[hi].skeleton[:, 0],
                        "-", color=col, linewidth=2.0, alpha=1.0)
        n_sel = sum(1 for hi in sol["selected_indices"]
                     if all_hypos[hi].frame == ft) if ft >= 0 else 0
        title = f"t={k}" if i > 0 else \
                f"seed={r['seed']}  t={k}  ({r['n_tracks']} tracks)"
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T-video", type=int, default=16)
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--sim-seed-start", type=int, default=4000)
    ap.add_argument("--ckpt", default=None,
                    help="use model rollouts if provided, else noisy-GT")
    ap.add_argument("--gt-noise-n", type=int, default=24)
    ap.add_argument("--gt-noise-angle-sigma", type=float, default=0.05)
    ap.add_argument("--gt-noise-step-sigma", type=float, default=0.3)
    ap.add_argument("--n-frames-per-row", type=int, default=8)
    ap.add_argument("--out-png", required=True)
    args = ap.parse_args()

    model = None
    if args.ckpt:
        params, ar_cfg = load_ckpt(args.ckpt)
        backbone = UNetARBackbone(cfg=ar_cfg)
        attach_head = AttachmentHead(cfg=ar_cfg)
        knot_gen = KnotGenerator(cfg=ar_cfg)
        sampler = make_sampler(ar_cfg, backbone, attach_head, knot_gen)
        model = dict(params=params, ar_cfg=ar_cfg, backbone=backbone,
                       attach_head=attach_head, knot_gen=knot_gen,
                       sampler=sampler)
        print(f"model: {args.ckpt}  (H={ar_cfg.H} T={ar_cfg.T} "
              f"n_knots={ar_cfg.n_knots} stop_head={ar_cfg.has_stop_head})")

    cfg = DiverseSimConfig(T=args.T_video, H=128, W=128,
                             sigma_scale_residual=False)
    build_cfg = BuildConfig(
        cost_mode="recon+score",
        pick_cost_base=5.0, score_bonus=100.0,
        max_pair_overlap_frac=0.3,
        birth_cost=200.0, death_cost=200.0,
        link_max_gap=2, link_max_dist=18.0,
        link_cost_scale=0.15, link_gap_cost_factor=1.5,
    )
    solve_cfg = SolveConfig(time_limit_s=15.0, num_workers=8)

    results = []
    for si in range(args.n_seeds):
        seed = args.sim_seed_start + si
        rng = np.random.default_rng(seed)
        r = track_one(seed, args.T_video, cfg, build_cfg, solve_cfg,
                        args.gt_noise_n, args.gt_noise_angle_sigma,
                        args.gt_noise_step_sigma, rng, model=model)
        if r is None:
            print(f"seed {seed}: no alive"); continue
        print(f"seed {seed}: n_alive={r['n_alive']} → {r['n_tracks']} tracks, "
              f"{len(r['sol']['selected_indices'])} picks")
        results.append(r)

    nrows = len(results)
    ncols = args.n_frames_per_row
    fig = plt.figure(figsize=(ncols * 2.4, nrows * 2.4))
    gs = fig.add_gridspec(nrows, ncols)
    for row_idx, r in enumerate(results):
        # Pick equally spaced frames from the anchor window
        af = r["anchor_frames"]
        idxs = np.linspace(0, len(af) - 1, ncols).astype(int)
        mid_frames = [af[i] for i in idxs]
        n_tracks = max(1, r["n_tracks"])
        rng = np.random.default_rng(row_idx)
        track_colors = rng.uniform(0.3, 1.0, size=(n_tracks, 3))
        render_row(fig, gs, row_idx, r, mid_frames, track_colors,
                     r["alive"], r["gt_curves"], r["all_hypos"], r["sol"],
                     r["hypo_track"], r["clip_raw"])
    cand_lbl = f"ckpt={Path(args.ckpt).name}" if args.ckpt else "noisy-GT candidates"
    fig.suptitle(
        f"Tracking on {nrows} sim seeds  ({cand_lbl}, "
        f"recon+score IP, T={args.T_video})\n"
        f"green = GT, colored = DIR selected (drawn on top of GT with black outline)",
        fontsize=10)
    fig.tight_layout()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
