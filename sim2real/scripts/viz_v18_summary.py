"""Clean summary viz: v18 champion on real (top half) + sim (bottom half)."""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2, jax, numpy as np, matplotlib.pyplot as plt

from sim2real.eval_v2.coverage import _chamfer_polylines, load_real_annotations
from sim2real.eval_v2.simlike_preproc import simlike_canonicalize
from sim2real.model.unet_ar import AttachmentHead, KnotGenerator, UNetARBackbone
from sim2real.scripts.ar_batched import make_sampler, sample_pool_one_clip
from sim2real.scripts.eval_ar import load_ckpt
from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-real", type=int, default=12)
    ap.add_argument("--n-sim", type=int, default=12)
    ap.add_argument("--ncols", type=int, default=6)
    ap.add_argument("--sim-seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params, cfg = load_ckpt(args.ckpt)
    backbone = UNetARBackbone(cfg=cfg)
    attach_head = AttachmentHead(cfg=cfg)
    knot_gen = KnotGenerator(cfg=cfg)
    sampler = make_sampler(cfg, backbone, attach_head, knot_gen)

    real_annots = load_real_annotations()[: args.n_real]
    sim_cfg = DiverseSimConfig(T=cfg.T, H=cfg.H, W=cfg.W,
                                 sigma_scale_residual=False)

    def process_real(ann, key):
        clip_r, smed_r, src_h, src_w, sy, sx = simlike_canonicalize(
            ann["meta"], T=cfg.T, target_hw=(cfg.H, cfg.W))
        gts = []
        for pl in ann["gt_polylines_native"]:
            if len(pl) >= 4:
                gts.append(pl.astype(np.float32) * np.asarray([sy, sx]))
        rollouts, key = sample_pool_one_clip(
            params, backbone, attach_head, knot_gen, cfg,
            clip_r, smed_r, [0.0, -8.0, 8.0], flips=(False, True),
            n_draws=2, n_attach=8, n_rollouts=4, score_thresh=0.02,
            key=key, _sampler=sampler)
        return smed_r, gts, rollouts, key

    def process_sim(sim_key):
        # Try up to 20 keys to find one with alive
        for _ in range(20):
            sim_key, k = jax.random.split(sim_key)
            out = sample_clip(k, sim_cfg)
            alive = np.asarray(out["flagella"]["alive"])
            if alive.sum() >= 1:
                break
        clip = np.asarray(out["clip_median"])
        smed = np.asarray(out["temporal_median"])
        curves = np.asarray(out["curves"][cfg.T // 2])
        gts = [curves[j] for j in range(curves.shape[0]) if bool(alive[j])]
        raw = np.asarray(out["clip_raw"][cfg.T // 2])
        rollouts, _ = sample_pool_one_clip(
            params, backbone, attach_head, knot_gen, cfg,
            clip, smed, [0.0, -8.0, 8.0], flips=(False, True),
            n_draws=2, n_attach=8, n_rollouts=4, score_thresh=0.02,
            key=jax.random.key(0), _sampler=sampler)
        return raw, gts, rollouts, sim_key

    key = jax.random.key(42)
    real_rows = []
    for ann in real_annots:
        try:
            smed, gts, rl, key = process_real(ann, key)
            real_rows.append((ann["name"], smed, gts, rl))
        except Exception as e:
            print(f"skip {ann['name']}: {e}")
    sim_rows = []
    sim_key = jax.random.key(args.sim_seed)
    for _ in range(args.n_sim):
        raw, gts, rl, sim_key = process_sim(sim_key)
        sim_rows.append((f"sim", raw, gts, rl))

    all_rows = ([("REAL", *r) for r in real_rows]
                 + [("SIM", *r) for r in sim_rows])
    nrows = (len(all_rows) + args.ncols - 1) // args.ncols
    fig, axes = plt.subplots(nrows, args.ncols,
                              figsize=(args.ncols * 2.4, nrows * 2.4),
                              squeeze=False)
    for i, row in enumerate(all_rows):
        r, c = i // args.ncols, i % args.ncols
        ax = axes[r][c]
        source, name, bg, gts, rollouts = row
        lo, hi = np.percentile(bg, [1, 99])
        gray = np.clip((bg - lo) / max(hi - lo, 1e-6), 0, 1)
        ax.imshow(gray, cmap="gray")
        # all rollouts as thin yellow
        for rl in rollouts:
            ax.plot(rl[:, 1], rl[:, 0], "-", color="#ffd633",
                    linewidth=0.5, alpha=0.10)
        # cyan best-per-GT rollout
        n_hit = 0
        for g in gts:
            if len(g) < 2: continue
            hit = False
            if rollouts:
                dists = [_chamfer_polylines(rl, g) for rl in rollouts]
                bi = int(np.argmin(dists))
                hit = dists[bi] <= 6.0
                brl = rollouts[bi]
                # cyan best line with dark outline
                ax.plot(brl[:, 1], brl[:, 0], "-", color="black",
                        linewidth=2.6, alpha=0.6)
                ax.plot(brl[:, 1], brl[:, 0], "-", color="#00e0ff",
                        linewidth=1.4, alpha=0.95)
            # GT line
            color = "#33ff44" if hit else "#ff3333"
            ax.plot(g[:, 1], g[:, 0], "-", color=color, linewidth=1.8,
                    alpha=0.85)
            if hit: n_hit += 1
        title_color = "#2222aa" if source == "REAL" else "#008833"
        ax.set_title(f"{source}: {name[-8:]}  {n_hit}/{len(gts)}",
                     fontsize=7, color=title_color)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, cfg.W - 1); ax.set_ylim(cfg.H - 1, 0)
    for i in range(len(all_rows), nrows * args.ncols):
        r, c = i // args.ncols, i % args.ncols
        axes[r][c].axis("off")

    fig.suptitle(f"v18 300k rollouts — REAL (top, blue titles) + SIM (bottom, green)\n"
                 f"yellow = all rollouts, cyan = best-per-GT, "
                 f"green GT = covered, red GT = miss",
                 fontsize=9)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
