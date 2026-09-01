"""Predictions on RAW real: display the canonical raw temporal median as
background, model predictions in yellow, GT in green.

Model runs on the 128×128 residual as usual; skeletons get scaled ×(256/128)
back to canonical 256 for display against the raw canonical background.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from sim2real.data.canonicalize import CANONICAL_H, CANONICAL_W
from sim2real.dir.run_dir import load_model_ckpt, load_pca_scaled
from sim2real.eval_v2.coverage import (
    _chamfer_polylines, canonicalize_real_frame, gt_polyline_to_canonical,
    load_real_annotations,
)
from sim2real.model.unet_energy import (
    UNetEnergy, decode_curves, sample_batched_noise, unpack_pred,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pca", required=True)
    ap.add_argument("--n-draws", type=int, default=8)
    ap.add_argument("--score-thresh", type=float, default=0.03)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--coverage-thresh", type=float, default=12.0,
                    help="in CANONICAL 256 px (equivalent to 6 at 128)")
    ap.add_argument("--n-clips", type=int, default=24)
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params, cfg_u = load_model_ckpt(args.ckpt)
    pca_mean, pca_basis = load_pca_scaled(args.pca)
    model = UNetEnergy(cfg=cfg_u)
    annots = load_real_annotations()[: args.n_clips]
    print(f"vizing {len(annots)} annotations at canonical {CANONICAL_H}x{CANONICAL_W}",
          flush=True)

    key = jax.random.key(0)
    rows = []
    for ai, ann in enumerate(annots):
        try:
            canon, cfg_can = canonicalize_real_frame(ann["meta"],
                                                       ann["src_width_px"],
                                                       T=cfg_u.T)
        except Exception as e:
            print(f"skip: {e}"); continue
        clip = canon["clip"]                                   # (T, 256, 256)
        static_median_canonical = canon["static_median"].astype(np.float32) / 255.0

        # Resize residual + median to model input size
        src_h, src_w = clip.shape[1], clip.shape[2]
        scale_y = cfg_u.H / src_h; scale_x = cfg_u.W / src_w
        clip_128 = np.stack([cv2.resize(clip[t], (cfg_u.W, cfg_u.H),
                                          interpolation=cv2.INTER_AREA)
                              for t in range(clip.shape[0])], axis=0).astype(np.float32)
        smed_128 = cv2.resize(static_median_canonical, (cfg_u.W, cfg_u.H),
                              interpolation=cv2.INTER_AREA).astype(np.float32)

        # GT in canonical 256 coords
        gt_canon = []
        for pl in ann["gt_polylines_native"]:
            g = gt_polyline_to_canonical(pl, ann["meta"], cfg_can,
                                          canonical_h=CANONICAL_H,
                                          canonical_w=CANONICAL_W)
            if len(g) >= 4: gt_canon.append(g)

        # Run model, get all candidates
        video = jnp.asarray(clip_128)[None]
        smed_in = jnp.asarray(smed_128)[None, ..., None]
        cands_128 = []
        for _ in range(args.n_draws):
            key, k = jax.random.split(key)
            noise = sample_batched_noise(k, 1, cfg_u)
            pred = model.apply(params, video, noise, smed_in, train=False)
            curves = np.asarray(decode_curves(pred, cfg_u, pca_mean, pca_basis))[0]
            f = unpack_pred(pred)
            s = np.asarray(jax.nn.sigmoid(f["score"][0])).ravel()
            flat_c = curves.reshape(-1, curves.shape[-2], 2)
            keep = np.where(s >= args.score_thresh)[0]
            if len(keep) > args.top_k:
                keep = keep[np.argsort(-s[keep])[:args.top_k]]
            for j in keep:
                cands_128.append(flat_c[j])
        # Scale candidates back to canonical 256 coords
        cands_canon = [c * np.asarray([1.0 / scale_y, 1.0 / scale_x])
                        for c in cands_128]

        rows.append(dict(name=ann["name"],
                          bg=static_median_canonical,
                          gt=gt_canon, cands=cands_canon))
        if len(rows) % 8 == 0:
            print(f"  processed {len(rows)}/{len(annots)}", flush=True)

    nrow = (len(rows) + args.ncols - 1) // args.ncols
    fig, axes = plt.subplots(nrow, args.ncols,
                              figsize=(args.ncols * 3.6, nrow * 3.4),
                              squeeze=False)
    for i, row in enumerate(rows):
        r, c = i // args.ncols, i % args.ncols
        ax = axes[r][c]
        bg = row["bg"]
        lo, hi = np.percentile(bg, [1, 99])
        gray = np.clip((bg - lo) / max(hi - lo, 1e-6), 0, 1)
        ax.imshow(gray, cmap="gray")
        n_hit = 0
        for gt in row["gt"]:
            if not row["cands"]:
                continue
            dists = [_chamfer_polylines(cd, gt) for cd in row["cands"]]
            best_idx = int(np.argmin(dists))
            best = row["cands"][best_idx]; best_dist = dists[best_idx]
            hit = best_dist <= args.coverage_thresh
            if hit: n_hit += 1
            color = "#33dd33" if hit else "#dd3333"
            ax.plot(best[:, 1], best[:, 0], "-", color="#ffcc00",
                    linewidth=1.8, alpha=0.9)
            ax.plot(gt[:, 1], gt[:, 0], "-", color=color, linewidth=2.0)
            ax.text(gt[0, 1] + 3, gt[0, 0] - 3, f"d={best_dist:.1f}",
                    fontsize=7, color=color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                              alpha=0.7))
        n_gt = len(row["gt"])
        ax.set_title(f"{row['name']}  {n_hit}/{n_gt} @ {args.coverage_thresh:.0f}px",
                     fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, CANONICAL_W - 1); ax.set_ylim(CANONICAL_H - 1, 0)

    for i in range(len(rows), nrow * args.ncols):
        r, c = i // args.ncols, i % args.ncols
        axes[r][c].axis("off")

    fig.suptitle(
        f"Real canonical raw median + model predictions   "
        f"green GT=hit, red GT=miss, yellow=closest candidate",
        fontsize=10)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
