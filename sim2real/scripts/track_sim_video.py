"""Track a T=16 sim video with v14 (or similar) via ILP over per-frame
rollouts + temporal links.

Pipeline:
  1. Sample a T=16 sim scene → clip_raw (16, H, W), gt curves per frame.
  2. Slide a T=4 window across frames 2..13 (12 anchor frames).
  3. For each anchor, run the AR model on the window → rollouts (each = a
     spline / polyline in image coords).
  4. Wrap each rollout as a Hypothesis(frame=k, ...) and feed to the
     existing sim2real/dir/build_problem + solve_problem.
  5. Render an mp4:
       - background: raw image at each frame
       - GT curves: green
       - selected DIR splines: cyan (linked across frames by track id)
       - all rollouts: yellow (thin, alpha)
"""
from __future__ import annotations
import argparse, pickle, time
from pathlib import Path

import cv2, jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt
import matplotlib.animation as animation

from sim2real.dir.build_problem import BuildConfig, Hypothesis, build_problem
from sim2real.dir.solve import SolveConfig, solve_problem
from sim2real.model.unet_ar import (
    AttachmentHead, KnotGenerator, UNetARBackbone, UNetARConfig,
)
from sim2real.scripts.ar_batched import make_sampler, sample_pool_one_clip
from sim2real.scripts.eval_ar import load_ckpt
from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip


def make_angle_noisy_candidates(gt_curve: np.ndarray, n_samples: int,
                                 angle_sigma: float, step_sigma: float,
                                 rng: np.random.Generator) -> list[np.ndarray]:
    """Generate rollout-style noisy versions of a GT curve.

    Instead of iid Gaussian on knot positions, perturb the per-step tangent
    angle and step length — matches how an AR model would sample small
    variations. Errors compound across knots as in a real rollout."""
    diffs = np.diff(gt_curve, axis=0)                          # (K, 2)
    step_lens = np.linalg.norm(diffs, axis=1)                  # (K,)
    tangents = np.arctan2(diffs[:, 0], diffs[:, 1])            # (K,) atan2(dy, dx)
    K = tangents.shape[0]
    d_ang = np.concatenate([[tangents[0]], np.diff(tangents)])
    out = []
    for _ in range(n_samples):
        n_ang = rng.normal(0, angle_sigma, K)
        n_step = rng.normal(0, step_sigma, K)
        d_ang_p = d_ang + n_ang
        step_p = np.clip(step_lens + n_step, 0.3, None)
        tan_p = np.cumsum(d_ang_p)
        curve = [gt_curve[0].copy()]
        pos = gt_curve[0].copy()
        for k in range(K):
            pos = pos + step_p[k] * np.array([np.sin(tan_p[k]),
                                                np.cos(tan_p[k])])
            curve.append(pos.copy())
        out.append(np.array(curve, dtype=np.float32))
    return out


def track_one_scene(ckpt_path: str, out_mp4: str, out_png: str,
                      T_video: int = 16, sim_seed: int = 2026,
                      n_draws: int = 2, n_attach: int = 8, n_rollouts: int = 4,
                      tta_angles=(0.0, -8.0, 8.0), score_thresh: float = 0.02,
                      pool_cap: int = 30,
                      candidate_source: str = "model",
                      gt_noise_n: int = 24, gt_noise_angle_sigma: float = 0.05,
                      gt_noise_step_sigma: float = 0.3):
    params, cfg = load_ckpt(ckpt_path)
    backbone = UNetARBackbone(cfg=cfg)
    attach_head = AttachmentHead(cfg=cfg)
    knot_gen = KnotGenerator(cfg=cfg)
    sampler = make_sampler(cfg, backbone, attach_head, knot_gen)
    T_win = cfg.T                                # e.g. 4

    # ---- Sample a T_video-long scene ---------------------------------------
    sim_cfg = DiverseSimConfig(T=T_video, H=cfg.H, W=cfg.W,
                                 sigma_scale_residual=False)
    out = sample_clip(jax.random.key(sim_seed), sim_cfg)
    clip_raw = np.asarray(out["clip_raw"]).astype(np.float32)     # (T, H, W)
    gt_curves = np.asarray(out["curves"])                         # (T, N, K, 2)
    alive = np.asarray(out["flagella"]["alive"])                  # (N,)
    gt_widths = np.asarray(out["flagella"]["width"])              # (N,)
    gt_amps = np.asarray(out["flagella"]["amp"])                  # (N,) signed
    n_alive = int(alive.sum())
    # Median amp/width per-rollout defaults (model doesn't predict them).
    if n_alive > 0:
        rollout_width = float(np.median(gt_widths[alive]))
        rollout_amp = float(np.median(gt_amps[alive]))
    else:
        rollout_width, rollout_amp = 1.5, -0.15
    print(f"scene: T={T_video} H={cfg.H}, {n_alive} alive flagella; "
          f"rollout width={rollout_width:.2f} amp={rollout_amp:.2f}")

    # ---- Slide a T_win window, run model at each anchor ---------------------
    key = jax.random.key(0)
    anchor_frames = list(range(T_win // 2, T_video - T_win // 2))
    all_hypos: list[Hypothesis] = []
    per_frame_rollouts: dict[int, list[np.ndarray]] = {}
    per_frame_scores: dict[int, list[float]] = {}
    rng = np.random.default_rng(0)
    t0 = time.time()
    if candidate_source == "noisy-gt":
        # Skip the model entirely; build candidates from GT + rollout-style noise
        for k in anchor_frames:
            per_frame_rollouts[k] = []
            per_frame_scores[k] = []
            for j in range(gt_curves.shape[1]):
                if not bool(alive[j]): continue
                g = gt_curves[k, j]
                cands = make_angle_noisy_candidates(g, gt_noise_n,
                                                     gt_noise_angle_sigma,
                                                     gt_noise_step_sigma, rng)
                for c in cands:
                    per_frame_rollouts[k].append(c)
                    per_frame_scores[k].append(0.9)
                    all_hypos.append(Hypothesis(
                        frame=k, skeleton=c.astype(np.float32),
                        width=float(gt_widths[j]), amp=float(gt_amps[j]),
                        score=0.9))
        print(f"noisy-GT: {len(all_hypos)} candidates across "
              f"{len(anchor_frames)} frames ({gt_noise_n} per GT per frame)")
    else:
        for k in anchor_frames:
            # Window [k - T_win//2, k + T_win - T_win//2)
            i0 = k - T_win // 2; i1 = i0 + T_win
            win_raw = clip_raw[i0:i1]
            med = np.median(win_raw, axis=0)
            win_median = win_raw - med
            smed = med.astype(np.float32)

            rollouts, scores, key = sample_pool_one_clip(
                params, backbone, attach_head, knot_gen, cfg,
                win_median.astype(np.float32), smed,
                list(tta_angles), flips=(False, True),
                n_draws=n_draws, n_attach=n_attach, n_rollouts=n_rollouts,
                score_thresh=score_thresh, key=key, _sampler=sampler,
                return_scores=True)

            if len(rollouts) > pool_cap:
                order = np.argsort(-np.asarray(scores))[:pool_cap]
                rollouts = [rollouts[i] for i in order]
                scores   = [scores[i] for i in order]

            per_frame_rollouts[k] = rollouts
            per_frame_scores[k] = scores
            for rl, sc in zip(rollouts, scores):
                all_hypos.append(Hypothesis(
                    frame=k, skeleton=rl.astype(np.float32),
                    width=rollout_width, amp=rollout_amp, score=float(sc)))
    print(f"model: {len(all_hypos)} hypotheses across {len(anchor_frames)} frames"
          f" in {time.time()-t0:.1f}s")

    # ---- Build ILP + solve --------------------------------------------------
    # Residuals per anchor frame (as v13's post-DIR eval used)
    T_active = len(anchor_frames)
    residuals = np.zeros((T_active, cfg.H, cfg.W), dtype=np.float32)
    frame_to_active = {k: i for i, k in enumerate(anchor_frames)}
    for k in anchor_frames:
        residuals[frame_to_active[k]] = clip_raw[k] - np.median(clip_raw, axis=0)
    # Rewrite frames in hypotheses to compressed ILP indices
    for h in all_hypos:
        h.frame = frame_to_active[h.frame]

    # Cost design for tracking:
    # - We want ~1 track per real flagellum (2 tracks here).
    # - Each pick costs `pick_cost_base` unless linked into a track (link
    #   cost < pick alone). Birth/death paid only at track endpoints so
    #   long chains amortize the birth+death across many picks.
    # - pool_cap=30 keeps ~30 candidates per frame → picks per frame ~= real
    #   count when overlap_frac is strict enough.
    # Cost design for tracking: each pick pays birth+death unless amortized
    # by links → long chains preferred. score_bonus makes picks attractive;
    # birth/death makes short/isolated picks expensive.
    # recon+score cost mode: cost = (residual_L1 after − before) − score·bonus
    # Rewards hypotheses whose Gaussian tube (width, amp) actually darkens
    # the residual where the flagellum is. Combined with per-rollout scores
    # this makes distractor tubes-through-noise unprofitable.
    build_cfg = BuildConfig(
        cost_mode="recon+score",
        pick_cost_base=5.0,
        score_bonus=100.0,
        max_pair_overlap_frac=0.3,
        birth_cost=200.0, death_cost=200.0,
        link_max_gap=2, link_max_dist=18.0,
        link_cost_scale=0.15, link_gap_cost_factor=1.5,
    )
    solve_cfg = SolveConfig(time_limit_s=30.0, num_workers=8)

    problem = build_problem(all_hypos, residuals, build_cfg)
    print(f"ILP: {problem['num_variables']} vars, "
          f"{len(problem['at_most_one_constraints'])} amo, "
          f"{len(problem['links'])} links")
    t1 = time.time()
    sol = solve_problem(problem, solve_cfg)
    print(f"solved: {sol['status']}  obj={sol['objective']:.1f}  "
          f"selected={len(sol['selected_indices'])}  tracks={len(sol['tracks'])}"
          f"  wall={sol['wall_time']:.1f}s")

    # ---- Build per-frame selected-with-track-id -----------------------------
    # tracks: list of lists of hypothesis indices (chain across frames)
    hypo_track_id = [-1] * len(all_hypos)
    for tid, track in enumerate(sol["tracks"]):
        for hi in track:
            hypo_track_id[hi] = tid

    # Stable color per track id
    n_tracks = max(1, len(sol["tracks"]))
    rng = np.random.default_rng(0)
    track_colors = rng.uniform(0.3, 1.0, size=(n_tracks, 3))

    # ---- Render summary PNG (grid of anchor frames) -------------------------
    ncols = 4
    nrows = (len(anchor_frames) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.5, nrows * 3.5), squeeze=False)
    for i, k in enumerate(anchor_frames):
        ax = axes[i // ncols][i % ncols]
        img = clip_raw[k]
        lo, hi = np.percentile(img, [1, 99])
        gray = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        ax.imshow(gray, cmap="gray")
        # GT (bright green, thick + shadow for contrast)
        for j in range(gt_curves.shape[1]):
            if not bool(alive[j]): continue
            g = gt_curves[k, j]
            ax.plot(g[:, 1], g[:, 0], "-", color="black", linewidth=3.5,
                    alpha=0.7)
            ax.plot(g[:, 1], g[:, 0], "-", color="#33ff44", linewidth=1.8,
                    alpha=1.0)
        # All candidates (yellow, more visible now)
        for rl in per_frame_rollouts.get(k, []):
            ax.plot(rl[:, 1], rl[:, 0], "-", color="#ffd633", linewidth=0.6,
                    alpha=0.30)
        # Selected DIR splines, colored by track id
        for hi in sol["selected_indices"]:
            h = all_hypos[hi]
            if h.frame != frame_to_active[k]: continue
            col = track_colors[hypo_track_id[hi]]
            ax.plot(h.skeleton[:, 1], h.skeleton[:, 0], "-", color=col,
                    linewidth=1.5, alpha=0.95)
        ax.set_title(f"t={k}   sel={len([hi for hi in sol['selected_indices'] if all_hypos[hi].frame == frame_to_active[k]])}",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for i in range(len(anchor_frames), nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    fig.suptitle(f"DIR tracking on sim (n_tracks={len(sol['tracks'])})",
                 fontsize=10)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")

    # ---- Render mp4 (or gif) ------------------------------------------------
    fig2, ax = plt.subplots(figsize=(6, 6))
    def frame_render(t):
        ax.clear()
        img = clip_raw[t]
        lo, hi = np.percentile(img, [1, 99])
        gray = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        ax.imshow(gray, cmap="gray")
        for j in range(gt_curves.shape[1]):
            if not bool(alive[j]): continue
            g = gt_curves[t, j]
            ax.plot(g[:, 1], g[:, 0], "-", color="#33ff44", linewidth=2.0,
                    alpha=0.9, label="GT" if j == 0 else None)
        if t in per_frame_rollouts:
            for rl in per_frame_rollouts[t]:
                ax.plot(rl[:, 1], rl[:, 0], "-", color="#ffd633", linewidth=0.5,
                        alpha=0.08)
        if t in frame_to_active:
            ft = frame_to_active[t]
            for hi in sol["selected_indices"]:
                h = all_hypos[hi]
                if h.frame != ft: continue
                col = track_colors[hypo_track_id[hi]]
                ax.plot(h.skeleton[:, 1], h.skeleton[:, 0], "-", color=col,
                        linewidth=2.2, alpha=0.95)
        ax.set_title(f"frame {t}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        return []

    ani = animation.FuncAnimation(fig2, frame_render, frames=T_video,
                                    interval=200, blit=False)
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    # Try mp4; fall back to gif
    try:
        ani.save(out_mp4, writer="ffmpeg", fps=4)
        print(f"wrote {out_mp4}")
    except Exception as e:
        gif = out_mp4.replace(".mp4", ".gif")
        ani.save(gif, writer="pillow", fps=4)
        print(f"wrote {gif} (mp4 failed: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--T-video", type=int, default=16)
    ap.add_argument("--sim-seed", type=int, default=2026)
    ap.add_argument("--candidate-source", choices=["model", "noisy-gt"],
                    default="model",
                    help="'noisy-gt' bypasses the model and uses rollout-style noisy GT")
    ap.add_argument("--gt-noise-n", type=int, default=24)
    ap.add_argument("--gt-noise-angle-sigma", type=float, default=0.05)
    ap.add_argument("--gt-noise-step-sigma", type=float, default=0.3)
    ap.add_argument("--out-mp4", required=True)
    ap.add_argument("--out-png", required=True)
    args = ap.parse_args()
    track_one_scene(args.ckpt, args.out_mp4, args.out_png,
                      T_video=args.T_video, sim_seed=args.sim_seed,
                      candidate_source=args.candidate_source,
                      gt_noise_n=args.gt_noise_n,
                      gt_noise_angle_sigma=args.gt_noise_angle_sigma,
                      gt_noise_step_sigma=args.gt_noise_step_sigma)


if __name__ == "__main__":
    main()
