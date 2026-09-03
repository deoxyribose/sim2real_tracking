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
from sim2real.eval.fg_ari import fg_ari
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
    ap.add_argument("--z-what-dim", type=int, default=64)
    ap.add_argument("--stem-channels", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--bg-base-res", type=int, default=4)
    ap.add_argument("--bg-channels", type=int, nargs="+", default=[8])
    ap.add_argument("--stem-strides", type=int, nargs="+", default=[2, 2, 2])
    ap.add_argument("--pres-hidden", type=int, default=128)
    ap.add_argument("--pres-depth", type=int, default=1)
    ap.add_argument("--pres-image-attn", action="store_true")
    ap.add_argument("--slot-competing-cross", action="store_true")
    ap.add_argument("--anchor-slots", action="store_true")
    ap.add_argument("--use-slot-attention", action="store_true")
    ap.add_argument("--use-neural-em", action="store_true")
    ap.add_argument("--use-isa", action="store_true",
                    help="Invariant Slot Attention (see pretrain.py --use-isa).")
    ap.add_argument("--isa-abs-pe-weight", type=float, default=0.0)
    ap.add_argument("--isa-anchor-init", action="store_true")
    ap.add_argument("--anchor-init-fixed", action="store_true")
    ap.add_argument("--z-what-init-std", type=float, default=0.2)
    ap.add_argument("--nem-attn-temp", type=float, default=1.0)
    ap.add_argument("--nem-use-bg-slot", action="store_true")
    ap.add_argument("--nem-bg-slot-per-pixel", action="store_true")
    ap.add_argument("--nem-dual-source-what", action="store_true")
    ap.add_argument("--nem-use-pca-theta", action="store_true")
    ap.add_argument("--decoder-use-film", action="store_true")
    ap.add_argument("--nem-theta-delta-scale", type=float, default=0.05)
    ap.add_argument("--hard-pres-gate", action="store_true",
                    help="Forbid dead-pred slots from winning alive-GT columns in the eval "
                         "matcher. Use if the model was trained with hard_pres_gate.")
    ap.add_argument("--savi-bootstrap", action="store_true",
                    help="Pass GT z_where[:, 0] as slot init per video. REQUIRED if the model "
                         "was trained with --savi-bootstrap, else it scores as untrained.")
    ap.add_argument("--encoder-type", default="vit", choices=["vit", "unet"])
    ap.add_argument("--unet-down-channels", type=int, nargs="+",
                    default=[24, 48, 96, 128, 192])
    ap.add_argument("--unet-n-up-stages", type=int, default=2)
    ap.add_argument("--unet-blocks-per-stage", type=int, default=2)
    args = ap.parse_args()

    ck = ckpt_load(args.ckpt)
    params = ck["params"]

    default_n_max = {"flagella": 8, "flagella_fast": 6, "many_cells": 48, "many_cells_fast": 24, "many_cells_small": 48, "easy_cells": 8, "two_cells": 4, "multiscale": 16, "worms": 12, "worms_fast": 8, "algae": 6, "algae_fast": 6}
    n_max = args.n_max if args.n_max is not None else default_n_max[args.sim]
    model_cfg = ModelConfig(
        n_max=n_max,
        d_model=args.d_model,
        n_heads=4,
        n_transformer_layers=args.n_transformer_layers,
        z_what_dim=args.z_what_dim,
        z_style_dim=args.z_style_dim,
        glimpse_size=args.glimpse_size,
        stem_channels=tuple(args.stem_channels),
        stem_strides=tuple(args.stem_strides),
        n_groups=args.n_groups,
        use_background=True,
        bg_base_res=args.bg_base_res,
        bg_channels=tuple(args.bg_channels),
        pres_hidden=args.pres_hidden,
        pres_depth=args.pres_depth,
        pres_image_attn=args.pres_image_attn,
        slot_competing_cross=args.slot_competing_cross,
        anchor_slots=args.anchor_slots,
        use_slot_attention=args.use_slot_attention,
        use_neural_em=args.use_neural_em,
        use_isa=args.use_isa,
        isa_abs_pe_weight=args.isa_abs_pe_weight,
        isa_anchor_init=args.isa_anchor_init,
        anchor_init_fixed=args.anchor_init_fixed,
        z_what_init_std=args.z_what_init_std,
        nem_attn_temp=args.nem_attn_temp,
        nem_use_bg_slot=args.nem_use_bg_slot,
        nem_bg_slot_per_pixel=args.nem_bg_slot_per_pixel,
        nem_dual_source_what=args.nem_dual_source_what,
        nem_use_pca_theta=args.nem_use_pca_theta,
        decoder_use_film=args.decoder_use_film,
        nem_theta_delta_scale=args.nem_theta_delta_scale,
        encoder_type=args.encoder_type,
        unet_down_channels=tuple(args.unet_down_channels),
        unet_n_up_stages=args.unet_n_up_stages,
        unet_blocks_per_stage=args.unet_blocks_per_stage,
    )
    model = SlotVideoModel(cfg=model_cfg)
    batch_fn, _ = build_sim(args.sim)
    key = jax.random.key(0)
    batch = batch_fn(key, args.batch)
    batch = slice_to_model(batch, n_max)

    boot0 = batch.z_where[:, 0] if args.savi_bootstrap else None

    @jax.jit
    def forward(v, k, b):
        return model.apply(params, v, k, bootstrap_zwhere0=b)

    keys = jax.random.split(key, args.batch)
    in_axes = (0, 0, 0 if boot0 is not None else None)
    outs = jax.vmap(forward, in_axes=in_axes)(batch.video, keys, boot0)

    # Recon metrics
    psnr_vals = [psnr(outs.composite[i], batch.video[i]) for i in range(args.batch)]
    ssim_vals = [ssim_simple(outs.composite[i], batch.video[i]) for i in range(args.batch)]

    # FG-ARI (primary segmentation metric — Locatello 2020 / SAVi standard).
    pred_masks_np = np.asarray(outs.masks_pred)
    gt_masks_np = np.asarray(batch.masks)
    gt_pres_np = np.asarray(batch.z_pres)
    ari_vals = []
    for i in range(args.batch):
        v = fg_ari(pred_masks_np[i], gt_masks_np[i], gt_pres_np[i])
        if v == v:  # skip NaN
            ari_vals.append(v)

    # Legacy matched-seg-IoU (kept for continuity; see memory feedback-metric-fg-ari).
    iou_vals = []
    for i in range(args.batch):
        v = matched_seg_iou(
            outs.z_where[i], outs.masks_pred[i],
            batch.z_where[i], batch.masks[i], batch.z_pres[i],
            pred_pres=outs.z_pres[i],
            hard_pres_gate=args.hard_pres_gate,
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
        "fg_ari": float(np.mean(ari_vals)) if ari_vals else float("nan"),
        "seg_iou_legacy": float(np.mean(iou_vals)),
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
