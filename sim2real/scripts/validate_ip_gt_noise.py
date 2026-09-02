"""Validate the tracking IP by feeding it GT + noisy copies + distractors.

If the ILP recovers the ~2 real tracks cleanly from this controlled input,
we know the formulation is correct → any downstream tracking failure is in
the MODEL producing too many false positives, not in the IP itself.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import jax, numpy as np, matplotlib.pyplot as plt, matplotlib.animation as animation

from sim2real.dir.build_problem import BuildConfig, Hypothesis, build_problem
from sim2real.dir.solve import SolveConfig, solve_problem
from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip


def noisy_copy(curve: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Perturb every knot by iid Gaussian(0, sigma)."""
    return curve + rng.normal(0, sigma, curve.shape).astype(np.float32)


def random_distractor(H: int, W: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """A totally random polyline: start at a random point, take K random steps."""
    y0, x0 = rng.uniform(20, H - 20), rng.uniform(20, W - 20)
    angle = rng.uniform(0, 2 * np.pi)
    knots = [(y0, x0)]
    for _ in range(K):
        angle += rng.normal(0, 0.4)
        step = rng.uniform(2, 5)
        y0 += step * np.sin(angle); x0 += step * np.cos(angle)
        knots.append((y0, x0))
    return np.array(knots, dtype=np.float32)


def run_multi_seed(args):
    """Sweep N sim seeds, report aggregate purity + track counts."""
    rows = []
    for si in range(args.n_seeds):
        seed = args.sim_seed + si
        cfg = DiverseSimConfig(T=args.T_video, H=128, W=128,
                                 sigma_scale_residual=False)
        out = sample_clip(jax.random.key(seed), cfg)
        clip_raw = np.asarray(out["clip_raw"])
        gt_curves = np.asarray(out["curves"])
        alive = np.asarray(out["flagella"]["alive"])
        gt_widths = np.asarray(out["flagella"]["width"])
        gt_amps = np.asarray(out["flagella"]["amp"])
        n_alive = int(alive.sum())
        if n_alive == 0:
            rows.append(dict(seed=seed, n_alive=0, skip=True))
            continue
        d_width = float(np.median(gt_widths[alive]))
        d_amp = float(np.median(gt_amps[alive]))
        rng = np.random.default_rng(seed)
        all_hypos, hypo_gt_flag = [], []
        for t in range(args.T_video):
            for j in range(gt_curves.shape[1]):
                if not bool(alive[j]): continue
                g = gt_curves[t, j]
                for _ in range(args.n_noisy_per_gt):
                    nc = noisy_copy(g, args.noise_sigma, rng)
                    all_hypos.append(Hypothesis(
                        frame=t, skeleton=nc.astype(np.float32),
                        width=float(gt_widths[j]), amp=float(gt_amps[j]),
                        score=float(0.9 - rng.uniform(0, 0.1))))
                    hypo_gt_flag.append(True)
            for _ in range(args.n_distractors_per_frame):
                d = random_distractor(cfg.H, cfg.W,
                                        gt_curves.shape[2] - 1, rng)
                all_hypos.append(Hypothesis(
                    frame=t, skeleton=d.astype(np.float32),
                    width=d_width, amp=d_amp,
                    score=float(0.3 + rng.uniform(0, 0.3))))
                hypo_gt_flag.append(False)
        residuals = np.zeros((args.T_video, cfg.H, cfg.W), dtype=np.float32)
        for t in range(args.T_video):
            residuals[t] = clip_raw[t] - np.median(clip_raw, axis=0)
        build_cfg = BuildConfig(
            cost_mode=args.cost_mode,
            pick_cost_base=args.pick_cost_base,
            score_bonus=args.score_bonus,
            max_pair_overlap_frac=args.overlap_frac,
            birth_cost=args.birth_cost, death_cost=args.death_cost,
            link_max_gap=args.link_max_gap, link_max_dist=args.link_max_dist,
            link_cost_scale=args.link_cost_scale, link_gap_cost_factor=1.5,
        )
        problem = build_problem(all_hypos, residuals, build_cfg)
        sol = solve_problem(problem, SolveConfig(time_limit_s=15.0, num_workers=8))
        sel_gt = sum(1 for i in sol["selected_indices"] if hypo_gt_flag[i])
        sel_di = sum(1 for i in sol["selected_indices"] if not hypo_gt_flag[i])
        # tracks with purity>=0.5 are "real"; ideally n_real == n_alive
        purities = []
        for tr in sol["tracks"]:
            p = sum(1 for i in tr if hypo_gt_flag[i]) / max(len(tr), 1)
            purities.append(p)
        n_real_tracks = sum(1 for p in purities if p >= 0.5)
        rows.append(dict(seed=seed, n_alive=n_alive,
                          n_tracks=len(sol["tracks"]),
                          n_real_tracks=n_real_tracks,
                          sel_gt=sel_gt, sel_di=sel_di,
                          skip=False))
        print(f"  seed {seed:4d}: n_alive={n_alive}  tracks={len(sol['tracks'])}"
              f"  real_tracks={n_real_tracks}  sel {sel_gt}GT+{sel_di}dist"
              f"  purities={[round(p, 2) for p in purities]}", flush=True)
    ok = [r for r in rows if not r["skip"]]
    if not ok:
        print("no valid seeds"); return
    n_alive_avg = np.mean([r["n_alive"] for r in ok])
    n_tracks_avg = np.mean([r["n_tracks"] for r in ok])
    n_real_avg = np.mean([r["n_real_tracks"] for r in ok])
    perfect = sum(1 for r in ok if r["n_real_tracks"] == r["n_alive"])
    print(f"\n{len(ok)}/{args.n_seeds} seeds valid; avg n_alive={n_alive_avg:.1f}, "
          f"tracks={n_tracks_avg:.1f}, real_tracks={n_real_avg:.1f}, "
          f"perfect={perfect}/{len(ok)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T-video", type=int, default=16)
    ap.add_argument("--sim-seed", type=int, default=2026)
    ap.add_argument("--n-seeds", type=int, default=1,
                    help=">1 for multi-seed aggregate stats (no viz)")
    ap.add_argument("--n-noisy-per-gt", type=int, default=6)
    ap.add_argument("--noise-sigma", type=float, default=1.5)
    ap.add_argument("--n-distractors-per-frame", type=int, default=20)
    ap.add_argument("--cost-mode", choices=["score_only", "recon+score"],
                    default="recon+score")
    ap.add_argument("--birth-cost", type=float, default=150.0)
    ap.add_argument("--death-cost", type=float, default=150.0)
    ap.add_argument("--score-bonus", type=float, default=100.0)
    ap.add_argument("--pick-cost-base", type=float, default=5.0)
    ap.add_argument("--overlap-frac", type=float, default=0.3)
    ap.add_argument("--link-max-dist", type=float, default=30.0)
    ap.add_argument("--link-max-gap", type=int, default=3)
    ap.add_argument("--link-cost-scale", type=float, default=0.03)
    ap.add_argument("--out-png", default=None)
    ap.add_argument("--out-mp4", default=None)
    args = ap.parse_args()

    if args.n_seeds > 1:
        run_multi_seed(args)
        return
    # ---- Sim scene ---------------------------------------------------------
    cfg = DiverseSimConfig(T=args.T_video, H=128, W=128,
                             sigma_scale_residual=False)
    out = sample_clip(jax.random.key(args.sim_seed), cfg)
    clip_raw = np.asarray(out["clip_raw"])                       # (T, H, W)
    gt_curves = np.asarray(out["curves"])                        # (T, N, K, 2)
    alive = np.asarray(out["flagella"]["alive"])                 # (N,)
    gt_widths = np.asarray(out["flagella"]["width"])             # (N,) per-slot width
    gt_amps = np.asarray(out["flagella"]["amp"])                 # (N,) per-slot signed amp
    n_alive = int(alive.sum())
    print(f"scene: T={args.T_video}, {n_alive} alive flagella")

    # ---- Build candidate pool ---------------------------------------------
    # For each frame, per alive GT, produce n_noisy copies. Plus random.
    # Amp/width for GT candidates come from sim's actual flagella params →
    # recon+score cost mode can evaluate them fairly against the residual.
    # Distractor amp/width use plausible defaults matched to sim medians.
    rng = np.random.default_rng(0)
    d_width = float(np.median(gt_widths[alive])) if n_alive else 1.5
    d_amp = float(np.median(gt_amps[alive])) if n_alive else -0.15
    all_hypos: list[Hypothesis] = []
    per_frame_rollouts: dict[int, list[np.ndarray]] = {}
    per_frame_gt_flags: dict[int, list[bool]] = {}
    for t in range(args.T_video):
        per_frame_rollouts[t] = []
        per_frame_gt_flags[t] = []
        for j in range(gt_curves.shape[1]):
            if not bool(alive[j]): continue
            g = gt_curves[t, j]
            for _ in range(args.n_noisy_per_gt):
                nc = noisy_copy(g, args.noise_sigma, rng)
                per_frame_rollouts[t].append(nc)
                per_frame_gt_flags[t].append(True)
                all_hypos.append(Hypothesis(
                    frame=t, skeleton=nc.astype(np.float32),
                    width=float(gt_widths[j]), amp=float(gt_amps[j]),
                    score=float(0.9 - rng.uniform(0, 0.1))))
        for _ in range(args.n_distractors_per_frame):
            d = random_distractor(cfg.H, cfg.W, gt_curves.shape[2] - 1, rng)
            per_frame_rollouts[t].append(d)
            per_frame_gt_flags[t].append(False)
            all_hypos.append(Hypothesis(
                frame=t, skeleton=d.astype(np.float32),
                width=d_width, amp=d_amp,
                score=float(0.3 + rng.uniform(0, 0.3))))
    print(f"pool: {len(all_hypos)} candidates "
          f"({n_alive * args.n_noisy_per_gt} GT-derived + "
          f"{args.n_distractors_per_frame} random) × {args.T_video} frames")

    # ---- ILP ---------------------------------------------------------------
    residuals = np.zeros((args.T_video, cfg.H, cfg.W), dtype=np.float32)
    for t in range(args.T_video):
        residuals[t] = clip_raw[t] - np.median(clip_raw, axis=0)

    build_cfg = BuildConfig(
        cost_mode=args.cost_mode,
        pick_cost_base=args.pick_cost_base,
        score_bonus=args.score_bonus,
        max_pair_overlap_frac=args.overlap_frac,
        birth_cost=args.birth_cost, death_cost=args.death_cost,
        link_max_gap=args.link_max_gap, link_max_dist=args.link_max_dist,
        link_cost_scale=args.link_cost_scale, link_gap_cost_factor=1.5,
    )
    solve_cfg = SolveConfig(time_limit_s=30.0, num_workers=8)

    problem = build_problem(all_hypos, residuals, build_cfg)
    print(f"ILP: {problem['num_variables']} vars, "
          f"{len(problem['at_most_one_constraints'])} amo, "
          f"{len(problem['links'])} links")
    sol = solve_problem(problem, solve_cfg)
    print(f"solved: {sol['status']}  obj={sol['objective']:.1f}  "
          f"selected={len(sol['selected_indices'])}  tracks={len(sol['tracks'])}"
          f"  wall={sol['wall_time']:.1f}s")

    # ---- Analyse: how many selected came from GT vs distractors? ----------
    gt_flag_arr = np.array([per_frame_gt_flags[h.frame][
        per_frame_rollouts[h.frame].index(h.skeleton.tolist()) if False else 0]
                              for h in all_hypos])   # (skip inversion; use hypo_gt_flag list directly)
    hypo_gt_flag = []
    i_local = {t: 0 for t in range(args.T_video)}
    for h in all_hypos:
        f = h.frame
        idx = i_local[f]
        hypo_gt_flag.append(per_frame_gt_flags[f][idx])
        i_local[f] += 1
    sel_gt = sum(1 for i in sol["selected_indices"] if hypo_gt_flag[i])
    sel_di = sum(1 for i in sol["selected_indices"] if not hypo_gt_flag[i])
    print(f"selected: {sel_gt} GT-derived  +  {sel_di} distractors")

    # Track purity: for each track, what fraction of picks are GT-derived?
    print("\ntrack purity (fraction of picks that came from GT):")
    for tid, tr in enumerate(sol["tracks"]):
        pure = sum(1 for i in tr if hypo_gt_flag[i]) / max(len(tr), 1)
        print(f"  track {tid}: L={len(tr):2d}  purity={pure:.2f}")

    # ---- Render summary + video ------------------------------------------
    n_tracks = max(1, len(sol["tracks"]))
    rng2 = np.random.default_rng(1)
    track_colors = rng2.uniform(0.3, 1.0, size=(n_tracks, 3))
    hypo_track = [-1] * len(all_hypos)
    for tid, tr in enumerate(sol["tracks"]):
        for hi in tr: hypo_track[hi] = tid

    ncols = 4
    nrows = (args.T_video + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.2, nrows * 3.2), squeeze=False)
    for t in range(args.T_video):
        ax = axes[t // ncols][t % ncols]
        img = clip_raw[t]
        lo, hi = np.percentile(img, [1, 99])
        gray = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        ax.imshow(gray, cmap="gray")
        # all candidates: GT-noise faint yellow, distractors faint red
        for k, rl in enumerate(per_frame_rollouts[t]):
            col = "#ffcc00" if per_frame_gt_flags[t][k] else "#ff8888"
            ax.plot(rl[:, 1], rl[:, 0], "-", color=col, linewidth=0.4,
                    alpha=0.25)
        # GT (thick green)
        for j in range(gt_curves.shape[1]):
            if not bool(alive[j]): continue
            g = gt_curves[t, j]
            ax.plot(g[:, 1], g[:, 0], "-", color="#33ff44", linewidth=2.0, alpha=0.95)
        # Selected — colored by track id
        for hi in sol["selected_indices"]:
            if all_hypos[hi].frame != t: continue
            col = track_colors[hypo_track[hi]]
            ax.plot(all_hypos[hi].skeleton[:, 1], all_hypos[hi].skeleton[:, 0],
                    "-", color=col, linewidth=1.6, alpha=0.95)
        n_sel = sum(1 for hi in sol["selected_indices"] if all_hypos[hi].frame == t)
        ax.set_title(f"t={t} sel={n_sel}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    for i in range(args.T_video, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    fig.suptitle(f"IP validation: GT + noise + distractors → "
                 f"{len(sol['tracks'])} tracks ({sel_gt}/{sel_gt+sel_di} GT-derived)",
                 fontsize=10)
    fig.tight_layout()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out_png}")

    # ---- mp4 ---------------------------------------------------------------
    fig2, ax = plt.subplots(figsize=(6, 6))
    def render(t):
        ax.clear()
        img = clip_raw[t]
        lo, hi = np.percentile(img, [1, 99])
        gray = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        ax.imshow(gray, cmap="gray")
        for k, rl in enumerate(per_frame_rollouts[t]):
            col = "#ffcc00" if per_frame_gt_flags[t][k] else "#ff8888"
            ax.plot(rl[:, 1], rl[:, 0], "-", color=col, linewidth=0.5, alpha=0.2)
        for j in range(gt_curves.shape[1]):
            if not bool(alive[j]): continue
            g = gt_curves[t, j]
            ax.plot(g[:, 1], g[:, 0], "-", color="black", linewidth=4, alpha=0.7)
            ax.plot(g[:, 1], g[:, 0], "-", color="#33ff44", linewidth=2.2, alpha=1.0)
        for hi in sol["selected_indices"]:
            if all_hypos[hi].frame != t: continue
            col = track_colors[hypo_track[hi]]
            ax.plot(all_hypos[hi].skeleton[:, 1], all_hypos[hi].skeleton[:, 0],
                    "-", color=col, linewidth=2.4, alpha=0.95)
        ax.set_title(f"t={t}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        return []
    ani = animation.FuncAnimation(fig2, render, frames=args.T_video,
                                    interval=250, blit=False)
    try:
        ani.save(args.out_mp4, writer="ffmpeg", fps=4)
        print(f"wrote {args.out_mp4}")
    except Exception as e:
        gif = args.out_mp4.replace(".mp4", ".gif")
        ani.save(gif, writer="pillow", fps=4)
        print(f"wrote {gif} (mp4 failed: {e})")


if __name__ == "__main__":
    main()
