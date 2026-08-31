"""Sanity: evaluate the model on held-out sim clips. If pre-DIR recall on sim
isn't ~100%, training is under-fit."""
from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from sim2real.dir.run_dir import load_model_ckpt, load_pca_scaled
from sim2real.eval_v2.coverage import _chamfer_polylines
from sim2real.model.unet_energy import (
    UNetEnergy, decode_curves, sample_batched_noise, unpack_pred,
)
from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pca", default="data_cache/flagella_pca.npz")
    ap.add_argument("--n-clips", type=int, default=200)
    ap.add_argument("--n-draws", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--score-thresh", type=float, default=0.03)
    ap.add_argument("--seed-offset", type=int, default=int(1e6),
                    help="Use a seed range disjoint from training")
    ap.add_argument("--coverage-thresh", type=float, default=12.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params, cfg_u = load_model_ckpt(args.ckpt)
    pca_mean, pca_basis = load_pca_scaled(args.pca)
    model = UNetEnergy(cfg=cfg_u)
    sim_cfg = DiverseSimConfig(T=cfg_u.T, H=cfg_u.H, W=cfg_u.W)

    all_recalls = []
    all_min_chamfers = []
    key = jax.random.key(0)
    for i in range(args.n_clips):
        out = sample_clip(jax.random.key(args.seed_offset + i), sim_cfg)
        gt_curves = np.asarray(out["curves"])[cfg_u.T // 2]
        alive = np.asarray(out["flagella"]["alive"])
        gt = [gt_curves[k] for k in range(gt_curves.shape[0]) if bool(alive[k])]
        if not gt:
            continue
        video = out["clip_median"][None]

        all_cands = []
        for _ in range(args.n_draws):
            key, k = jax.random.split(key)
            noise = sample_batched_noise(k, 1, cfg_u)
            pred = model.apply(params, jnp.asarray(video), noise, train=False)
            curves = np.asarray(decode_curves(pred, cfg_u, pca_mean, pca_basis))[0]
            f = unpack_pred(pred)
            s = np.asarray(jax.nn.sigmoid(f["score"][0])).ravel()
            flat_c = curves.reshape(-1, curves.shape[-2], 2)
            keep = np.where(s >= args.score_thresh)[0]
            if len(keep) > args.top_k:
                keep = keep[np.argsort(-s[keep])[:args.top_k]]
            for j in keep:
                all_cands.append(flat_c[j])

        for g in gt:
            best = min([_chamfer_polylines(c, g) for c in all_cands],
                        default=float("inf"))
            all_min_chamfers.append(best)
            all_recalls.append(best <= args.coverage_thresh)

        if (i + 1) % max(1, args.n_clips // 10) == 0:
            print(f"  [{i+1}/{args.n_clips}] running mean_recall={np.mean(all_recalls):.3f}",
                  flush=True)

    all_min_chamfers = np.asarray(all_min_chamfers)
    result = dict(
        n_gt_total=int(len(all_recalls)),
        pre_dir_recall=float(np.mean(all_recalls)),
        min_chamfer_p50=float(np.percentile(all_min_chamfers, 50)),
        min_chamfer_p90=float(np.percentile(all_min_chamfers, 90)),
        recall_at_thresholds={str(t): float((all_min_chamfers <= t).mean())
                                for t in [4, 6, 8, 10, 12, 16, 20]},
    )
    print(f"SIM SANITY   pre-DIR recall @ {args.coverage_thresh}px = "
          f"{result['pre_dir_recall']:.3f}")
    print(f"             chamfer p50 = {result['min_chamfer_p50']:.2f}   "
          f"p90 = {result['min_chamfer_p90']:.2f}")
    print(f"             by threshold: {result['recall_at_thresholds']}")
    open(args.out, "w").write(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
