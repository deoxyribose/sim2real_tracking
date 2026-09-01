"""Honest recall viz: for each GT flagellum, show only the CLOSEST candidate
by Chamfer distance. If the closest is within threshold, that GT is
"recalled". This is exactly what the recall metric measures — no visual
inflation from unrelated selected candidates."""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from sim2real.dir.run_dir import load_model_ckpt, load_pca_scaled
from sim2real.eval_v2.coverage import (
    _chamfer_polylines, canonicalize_real_frame, gt_polyline_to_canonical,
    load_real_annotations,
)
from sim2real.model.unet_energy import (
    UNetEnergy, decode_curves, sample_batched_noise, unpack_pred,
)
from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip


def rgb_signed(f, rng):
    return np.clip((f + rng) / (2 * rng), 0, 1)


def sample_candidates(params, model, cfg_u, video, key, n_draws, score_thresh,
                       top_k, pca_mean, pca_basis, static_median=None):
    smed = None if static_median is None else jnp.asarray(static_median)[None, ..., None]
    all_curves = []
    for _ in range(n_draws):
        key, k = jax.random.split(key)
        noise = sample_batched_noise(k, 1, cfg_u)
        pred = model.apply(params, video, noise, smed, train=False)
        curves = np.asarray(decode_curves(pred, cfg_u, pca_mean, pca_basis))[0]
        f = unpack_pred(pred)
        s = np.asarray(jax.nn.sigmoid(f["score"][0])).ravel()
        flat_c = curves.reshape(-1, curves.shape[-2], 2)
        keep = np.where(s >= score_thresh)[0]
        if len(keep) > top_k:
            keep = keep[np.argsort(-s[keep])[:top_k]]
        all_curves.extend([flat_c[j] for j in keep])
    return all_curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pca", default="data_cache/flagella_pca.npz")
    ap.add_argument("--n-draws", type=int, default=8)
    ap.add_argument("--score-thresh", type=float, default=0.03)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--coverage-thresh", type=float, default=12.0)
    ap.add_argument("--source", choices=["real", "sim"], required=True)
    ap.add_argument("--n-clips", type=int, default=16)
    ap.add_argument("--sim-seed-offset", type=int, default=int(1e6))
    ap.add_argument("--out", required=True)
    ap.add_argument("--show-raw", action="store_true",
                    help="Show clip_raw (full composite) instead of median-subtracted residual")
    args = ap.parse_args()

    params, cfg_u = load_model_ckpt(args.ckpt)
    pca_mean, pca_basis = load_pca_scaled(args.pca)
    model = UNetEnergy(cfg=cfg_u)

    rows = []
    key = jax.random.key(0)
    if args.source == "sim":
        sim_cfg = DiverseSimConfig(T=cfg_u.T, H=cfg_u.H, W=cfg_u.W)
        for i in range(args.n_clips):
            out = sample_clip(jax.random.key(args.sim_seed_offset + i), sim_cfg)
            residual = np.asarray(out["clip_median"])
            display_clip = np.asarray(out["clip_raw"]) if args.show_raw else residual
            gt_curves = np.asarray(out["curves"])[cfg_u.T // 2]
            alive = np.asarray(out["flagella"]["alive"])
            gt_valid = [gt_curves[k] for k in range(gt_curves.shape[0])
                         if bool(alive[k])]
            key, kk = jax.random.split(key)
            smed = np.asarray(out["temporal_median"])
            cands = sample_candidates(params, model, cfg_u,
                                        jnp.asarray(residual)[None], kk,
                                        args.n_draws, args.score_thresh,
                                        args.top_k, pca_mean, pca_basis,
                                        static_median=smed)
            rows.append(dict(name=f"sim seed={args.sim_seed_offset + i}",
                              residual=display_clip, gt=gt_valid, cands=cands))
    else:
        annots = load_real_annotations()[: args.n_clips]
        for ann in annots:
            try:
                canon, cfg_can = canonicalize_real_frame(ann["meta"],
                                                           ann["src_width_px"],
                                                           T=cfg_u.T)
            except Exception:
                continue
            clip = canon["clip"]
            src_h, src_w = clip.shape[1], clip.shape[2]
            scale_y = scale_x = 1.0
            if src_h != cfg_u.H or src_w != cfg_u.W:
                import cv2
                scale_y = cfg_u.H / src_h; scale_x = cfg_u.W / src_w
                clip = np.stack([cv2.resize(clip[t], (cfg_u.W, cfg_u.H),
                                              interpolation=cv2.INTER_AREA)
                                  for t in range(clip.shape[0])], axis=0).astype(np.float32)
            from sim2real.data.canonicalize import CANONICAL_H, CANONICAL_W
            gt = []
            for pl_native in ann["gt_polylines_native"]:
                gc = gt_polyline_to_canonical(pl_native, ann["meta"], cfg_can,
                                                canonical_h=CANONICAL_H,
                                                canonical_w=CANONICAL_W)
                if len(gc) >= 4:
                    gt.append(gc * np.asarray([scale_y, scale_x]))
            key, kk = jax.random.split(key)
            smed = canon.get("static_median")
            if smed is None:
                smed = np.median(clip, axis=0).astype(np.float32)
            else:
                smed = smed.astype(np.float32) / 255.0
                if smed.shape[0] != cfg_u.H or smed.shape[1] != cfg_u.W:
                    import cv2
                    smed = cv2.resize(smed, (cfg_u.W, cfg_u.H),
                                       interpolation=cv2.INTER_AREA).astype(np.float32)
            cands = sample_candidates(params, model, cfg_u,
                                        jnp.asarray(clip)[None], kk,
                                        args.n_draws, args.score_thresh,
                                        args.top_k, pca_mean, pca_basis,
                                        static_median=smed)
            rows.append(dict(name=ann["name"], residual=clip, gt=gt, cands=cands))

    ncol = 2
    nrow = (len(rows) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 6, nrow * 6),
                              squeeze=False)
    for i, row in enumerate(rows):
        r, c = i // ncol, i % ncol
        ax = axes[r][c]
        t = row["residual"].shape[0] // 2
        frame = row["residual"][t]
        # If values are non-negative, treat as raw intensity (grayscale).
        # Otherwise treat as signed residual (seismic).
        if frame.min() >= -0.05:
            lo, hi = np.percentile(frame, [1, 99])
            gray = np.clip((frame - lo) / max(hi - lo, 1e-6), 0, 1)
            ax.imshow(gray, cmap="gray")
        else:
            rng = max(float(np.percentile(np.abs(frame), 99.5)), 0.02)
            ax.imshow(rgb_signed(frame, rng), cmap="seismic")

        n_hit = 0
        for gt in row["gt"]:
            if not row["cands"]:
                continue
            dists = [_chamfer_polylines(cand, gt) for cand in row["cands"]]
            best_idx = int(np.argmin(dists))
            best_dist = dists[best_idx]
            hit = best_dist <= args.coverage_thresh
            if hit: n_hit += 1
            best = row["cands"][best_idx]
            color = "#33dd33" if hit else "#dd3333"
            ax.plot(best[:, 1], best[:, 0], "-", color="#ffcc00",
                    linewidth=2.5, alpha=0.9)
            # GT on top in green (hit) or red-ish (miss)
            ax.plot(gt[:, 1], gt[:, 0], "-", color=color, linewidth=2.5)
            # dist label
            ax.text(gt[0, 1] + 4, gt[0, 0] - 4,
                    f"d={best_dist:.1f}", fontsize=8, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                              alpha=0.7))
        n_gt = len(row["gt"])
        ax.set_title(f"{row['name']}   {n_hit}/{n_gt} hits @ {args.coverage_thresh}px",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, cfg_u.W - 1); ax.set_ylim(cfg_u.H - 1, 0)

    for i in range(len(rows), nrow * ncol):
        r, c = i // ncol, i % ncol
        axes[r][c].axis("off")

    fig.suptitle(
        f"Honest recall viz — yellow = closest candidate to each GT, "
        f"green GT = hit within {args.coverage_thresh}px, red GT = miss",
        fontsize=10)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
