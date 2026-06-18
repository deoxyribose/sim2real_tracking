"""Evaluate a checkpoint on a simulator. Reports recon (PSNR/SSIM), seg IoU, latent probes.

Usage:
    PYTHONPATH=. python3 -m sim2real.scripts.eval_ckpt --ckpt runs/pretrain_many_cells/ckpts/step_500.pkl --sim many_cells
"""

from __future__ import annotations

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from sim2real.eval.disentangle import linear_probe_zwhere_from_zwhat, silhouette_zwhat
from sim2real.eval.recon import psnr, ssim_simple
from sim2real.eval.seg_iou import matched_seg_iou
from sim2real.eval.tra import id_switch_count
from sim2real.model.model import ModelConfig, SlotVideoModel
from sim2real.sim.api import build_sim
from sim2real.train.ckpt import load as ckpt_load


def slice_to_model(batch, Nm):
    from sim2real.types import SimSample
    return SimSample(
        video=batch.video,
        z_where=batch.z_where[:, :, :Nm],
        z_pres=batch.z_pres[:, :, :Nm],
        z_style=batch.z_style,
        masks=batch.masks[:, :, :Nm],
        z_what=None if batch.z_what is None else batch.z_what[:, :Nm],
        meta=batch.meta,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sim", default="flagella")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-max", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--z-style-dim", type=int, default=4)
    ap.add_argument("--glimpse-size", type=int, default=32)
    ap.add_argument("--n-groups", type=int, default=1)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-transformer-layers", type=int, default=2)
    ap.add_argument("--bg-base-res", type=int, default=4)
    ap.add_argument("--bg-channels", type=int, nargs="+", default=[8])
    args = ap.parse_args()

    ck = ckpt_load(args.ckpt)
    params = ck["params"]

    default_n_max = {"flagella": 8, "many_cells": 48, "multiscale": 16, "worms": 12}
    n_max = args.n_max if args.n_max is not None else default_n_max[args.sim]
    model_cfg = ModelConfig(
        n_max=n_max,
        d_model=args.d_model,
        n_heads=4,
        n_transformer_layers=args.n_transformer_layers,
        z_what_dim=64,
        z_style_dim=args.z_style_dim,
        glimpse_size=args.glimpse_size,
        stem_channels=(16, 32, 64),
        n_groups=args.n_groups,
        use_background=True,
        bg_base_res=args.bg_base_res,
        bg_channels=tuple(args.bg_channels),
    )
    model = SlotVideoModel(cfg=model_cfg)
    batch_fn, _ = build_sim(args.sim)
    key = jax.random.key(0)
    batch = batch_fn(key, args.batch)
    batch = slice_to_model(batch, n_max)

    forward = jax.jit(lambda v, k: model.apply(params, v, k))
    keys = jax.random.split(key, args.batch)
    outs = jax.vmap(forward)(batch.video, keys)

    # Recon metrics
    psnr_vals = [psnr(outs.composite[i], batch.video[i]) for i in range(args.batch)]
    ssim_vals = [ssim_simple(outs.composite[i], batch.video[i]) for i in range(args.batch)]

    # Seg IoU
    iou_vals = []
    for i in range(args.batch):
        v = matched_seg_iou(
            outs.z_where[i], outs.masks_pred[i],
            batch.z_where[i], batch.masks[i], batch.z_pres[i],
        )
        iou_vals.append(float(v))

    # Latent probes (use frame 0 GT identity = slot index, since slot order is identity).
    pred_zwhat = np.asarray(outs.z_what)             # (B, T, N, Zw)
    gt_zwhere = np.asarray(batch.z_where)            # (B, T, N, 3)
    gt_pres = np.asarray(batch.z_pres)               # (B, T, N)
    gt_ids = np.broadcast_to(
        np.arange(n_max)[None, None, :], gt_pres.shape
    )                                                # slot index as identity
    # Flatten batch into outer axis for probes
    pred_zwhat_flat = pred_zwhat.reshape(-1, *pred_zwhat.shape[2:])
    gt_zwhere_flat = gt_zwhere.reshape(-1, *gt_zwhere.shape[2:])
    gt_pres_flat = gt_pres.reshape(-1, *gt_pres.shape[2:])
    gt_ids_flat = gt_ids.reshape(-1, *gt_ids.shape[2:])

    silhouette = silhouette_zwhat(pred_zwhat_flat, gt_pres_flat, gt_ids_flat)
    probe_r2 = linear_probe_zwhere_from_zwhat(pred_zwhat_flat, gt_zwhere_flat, gt_pres_flat)

    # Tracking metric: ID switches summed across videos
    id_switches = 0
    for i in range(args.batch):
        id_switches += id_switch_count(outs.z_where[i], outs.z_what[i], batch.z_where[i], batch.z_pres[i])

    metrics = {
        "ckpt": args.ckpt,
        "sim": args.sim,
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "seg_iou": float(np.mean(iou_vals)),
        "silhouette_zwhat": silhouette,
        "linear_probe_r2_zwhere_from_zwhat": probe_r2,
        "id_switches_total": id_switches,
    }
    print(json.dumps(metrics, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
