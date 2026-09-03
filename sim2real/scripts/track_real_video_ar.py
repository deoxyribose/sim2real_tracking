"""Track flagella in a REAL video with an AR ckpt (v18-era).

Loads a T=20 raw real video around a labeled frame, runs the model with
sim-like preproc at each anchor, feeds rollouts to the tracking IP,
renders an mp4."""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2, jax, numpy as np, matplotlib.pyplot as plt, matplotlib.animation as animation

from sim2real.dir.build_problem import BuildConfig, Hypothesis, build_problem
from sim2real.dir.solve import SolveConfig, solve_problem
from sim2real.eval_v2.coverage import load_real_annotations, load_source_clip
from sim2real.model.unet_ar import AttachmentHead, KnotGenerator, UNetARBackbone
from sim2real.scripts.ar_batched import make_sampler, sample_pool_one_clip
from sim2real.scripts.eval_ar import load_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ann-idx", type=int, default=8)
    ap.add_argument("--T-video", type=int, default=20)
    ap.add_argument("--pool-cap", type=int, default=30)
    ap.add_argument("--out-mp4", required=True)
    ap.add_argument("--out-png", required=True)
    args = ap.parse_args()

    params, cfg = load_ckpt(args.ckpt)
    backbone = UNetARBackbone(cfg=cfg); attach_head = AttachmentHead(cfg=cfg)
    knot_gen = KnotGenerator(cfg=cfg)
    sampler = make_sampler(cfg, backbone, attach_head, knot_gen)
    T_win = cfg.T

    ann = load_real_annotations()[args.ann_idx]
    print(f"annotation: {ann['name']} (source={ann['source']})")
    raw = load_source_clip(ann["source"], ann["meta"], T=args.T_video)
    raw = raw.astype(np.float32) / 255.0
    src_h, src_w = raw.shape[1], raw.shape[2]
    raw_r = np.stack([cv2.resize(raw[t], (cfg.W, cfg.H),
                                    interpolation=cv2.INTER_AREA)
                        for t in range(raw.shape[0])], axis=0).astype(np.float32)

    anchor_frames = list(range(T_win // 2, args.T_video - T_win // 2))
    all_hypos: list[Hypothesis] = []
    per_frame_rollouts = {}
    key = jax.random.key(0)
    for k in anchor_frames:
        i0 = k - T_win // 2; i1 = i0 + T_win
        win_raw = raw_r[i0:i1]
        med = np.median(win_raw, axis=0)
        clip = win_raw - med
        rollouts, scores, key = sample_pool_one_clip(
            params, backbone, attach_head, knot_gen, cfg,
            clip.astype(np.float32), med.astype(np.float32),
            [0.0, -8.0, 8.0], flips=(False, True),
            n_draws=2, n_attach=8, n_rollouts=4,
            score_thresh=0.02, key=key, _sampler=sampler,
            return_scores=True)
        if len(rollouts) > args.pool_cap:
            order = np.argsort(-np.asarray(scores))[:args.pool_cap]
            rollouts = [rollouts[i] for i in order]
            scores = [scores[i] for i in order]
        per_frame_rollouts[k] = rollouts
        for rl, sc in zip(rollouts, scores):
            all_hypos.append(Hypothesis(
                frame=k, skeleton=rl.astype(np.float32),
                width=1.2, amp=-0.05, score=float(sc)))
    print(f"pool: {len(all_hypos)} candidates across {len(anchor_frames)} frames")

    residuals = np.zeros((len(anchor_frames), cfg.H, cfg.W), dtype=np.float32)
    for i, k in enumerate(anchor_frames):
        residuals[i] = raw_r[k] - np.median(raw_r, axis=0)
    for h in all_hypos:
        h.frame = anchor_frames.index(h.frame)
    build_cfg = BuildConfig(
        cost_mode="recon+score",
        pick_cost_base=5.0, score_bonus=100.0,
        max_pair_overlap_frac=0.3,
        birth_cost=200.0, death_cost=200.0,
        link_max_gap=2, link_max_dist=18.0,
        link_cost_scale=0.15, link_gap_cost_factor=1.5,
    )
    solve_cfg = SolveConfig(time_limit_s=30.0, num_workers=8)
    problem = build_problem(all_hypos, residuals, build_cfg)
    sol = solve_problem(problem, solve_cfg)
    print(f"solved: {sol['status']}  selected={len(sol['selected_indices'])} "
          f"tracks={len(sol['tracks'])}")

    hypo_track = [-1] * len(all_hypos)
    for tid, tr in enumerate(sol["tracks"]):
        for hi in tr: hypo_track[hi] = tid
    n_tracks = max(1, len(sol["tracks"]))
    rng = np.random.default_rng(0)
    track_colors = rng.uniform(0.3, 1.0, size=(n_tracks, 3))

    ncols = 4
    nrows = (len(anchor_frames) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.4, nrows * 3.4), squeeze=False)
    for i, k in enumerate(anchor_frames):
        ax = axes[i // ncols][i % ncols]
        img = raw_r[k]
        lo, hi = np.percentile(img, [1, 99])
        gray = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        ax.imshow(gray, cmap="gray")
        for rl in per_frame_rollouts.get(k, []):
            ax.plot(rl[:, 1], rl[:, 0], "-", color="#ffd633",
                    linewidth=0.5, alpha=0.10)
        ft = anchor_frames.index(k)
        for hi_ in sol["selected_indices"]:
            if all_hypos[hi_].frame != ft: continue
            col = track_colors[hypo_track[hi_]]
            ax.plot(all_hypos[hi_].skeleton[:, 1], all_hypos[hi_].skeleton[:, 0],
                    "-", color="black", linewidth=3.0, alpha=0.7)
            ax.plot(all_hypos[hi_].skeleton[:, 1], all_hypos[hi_].skeleton[:, 0],
                    "-", color=col, linewidth=1.8, alpha=0.98)
        n_sel = sum(1 for hi_ in sol["selected_indices"]
                     if all_hypos[hi_].frame == ft)
        ax.set_title(f"t={k}  sel={n_sel}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for i in range(len(anchor_frames), nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    fig.suptitle(f"v18 tracking on REAL {ann['name']} ({len(sol['tracks'])} tracks)",
                 fontsize=10)
    fig.tight_layout()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out_png}")

    fig2, ax = plt.subplots(figsize=(6, 6))
    def render(t):
        ax.clear()
        img = raw_r[t]
        lo, hi = np.percentile(img, [1, 99])
        gray = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        ax.imshow(gray, cmap="gray")
        if t in per_frame_rollouts:
            for rl in per_frame_rollouts[t]:
                ax.plot(rl[:, 1], rl[:, 0], "-", color="#ffd633",
                        linewidth=0.5, alpha=0.10)
            ft = anchor_frames.index(t)
            for hi_ in sol["selected_indices"]:
                if all_hypos[hi_].frame != ft: continue
                col = track_colors[hypo_track[hi_]]
                ax.plot(all_hypos[hi_].skeleton[:, 1], all_hypos[hi_].skeleton[:, 0],
                        "-", color="black", linewidth=3.5, alpha=0.7)
                ax.plot(all_hypos[hi_].skeleton[:, 1], all_hypos[hi_].skeleton[:, 0],
                        "-", color=col, linewidth=2.2, alpha=0.98)
        ax.set_title(f"real {ann['name']}  t={t}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        return []
    ani = animation.FuncAnimation(fig2, render, frames=args.T_video,
                                    interval=200, blit=False)
    try:
        ani.save(args.out_mp4, writer="ffmpeg", fps=5)
        print(f"wrote {args.out_mp4}")
    except Exception as e:
        gif = args.out_mp4.replace(".mp4", ".gif")
        ani.save(gif, writer="pillow", fps=5)
        print(f"wrote {gif} (mp4 failed: {e})")


if __name__ == "__main__":
    main()
