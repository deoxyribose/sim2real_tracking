"""Measure systematic position + scale errors of alive pred slots vs matched GT cells.

For each video: Hungarian-match alive pred slots to alive GT slots by position, then report
the residual error distribution. If position error is bimodal or clustered near integer
grid multiples → sub-grid quantization. If scale error is systematically off → decoder bias.

Usage:
    PYTHONPATH=. python3 -m sim2real.scripts.position_error_diag \
        --ckpt runs/nem_temp01_50k/ckpts/step_50000.pkl \
        --sim many_cells_fast --n-max 18 \
        --glimpse-size 16 --d-model 128 --n-transformer-layers 3 --stem-strides 2 2 1 \
        --use-neural-em --anchor-init-fixed --z-what-init-std 1.0 --nem-attn-temp 0.1 \
        --n-videos 32
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

from sim2real.model.model import ModelConfig, SlotVideoModel
from sim2real.sim.api import build_sim
from sim2real.train.ckpt import load as ckpt_load
from sim2real.types import SimSample


def slice_to_model(batch, Nm):
    return SimSample(
        video=batch.video, z_where=batch.z_where[:, :, :Nm],
        z_pres=batch.z_pres[:, :, :Nm], z_style=batch.z_style,
        masks=batch.masks[:, :, :Nm],
        z_what=None if batch.z_what is None else batch.z_what[:, :Nm],
        meta=batch.meta,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--n-max", type=int, required=True)
    ap.add_argument("--n-videos", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-transformer-layers", type=int, default=3)
    ap.add_argument("--glimpse-size", type=int, default=16)
    ap.add_argument("--stem-strides", type=int, nargs="+", default=[2, 2, 1])
    ap.add_argument("--use-neural-em", action="store_true")
    ap.add_argument("--anchor-init-fixed", action="store_true")
    ap.add_argument("--z-what-init-std", type=float, default=0.2)
    ap.add_argument("--nem-attn-temp", type=float, default=1.0)
    ap.add_argument("--nem-use-bg-slot", action="store_true")
    args = ap.parse_args()

    ck = ckpt_load(args.ckpt)
    params = ck["params"]
    cfg = ModelConfig(
        n_max=args.n_max, d_model=args.d_model, n_heads=4,
        n_transformer_layers=args.n_transformer_layers,
        z_what_dim=64, z_style_dim=4, glimpse_size=args.glimpse_size,
        stem_channels=(16, 32, 64), stem_strides=tuple(args.stem_strides),
        n_groups=1, use_background=True, bg_base_res=4, bg_channels=(8,),
        use_neural_em=args.use_neural_em,
        anchor_init_fixed=args.anchor_init_fixed,
        z_what_init_std=args.z_what_init_std,
        nem_attn_temp=args.nem_attn_temp,
        nem_use_bg_slot=args.nem_use_bg_slot,
    )
    model = SlotVideoModel(cfg=cfg)
    batch_fn, _ = build_sim(args.sim)

    key = jax.random.key(args.seed)
    batch = jax.jit(lambda k: batch_fn(k, args.n_videos))(jax.random.split(key)[0])
    bm = slice_to_model(batch, args.n_max)
    T, N = bm.video.shape[1], args.n_max
    H, W = bm.video.shape[2], bm.video.shape[3]

    @jax.jit
    def fwd(video, k):
        keys = jax.random.split(k, video.shape[0])
        return jax.vmap(lambda v, kk: model.apply(params, v, kk))(video, keys)

    outs = fwd(bm.video, key)
    pred_zw = np.asarray(outs.z_where)                                                     # (B, T, N, 5)
    pred_zp = np.asarray(outs.z_pres)                                                      # (B, T, N)
    gt_zw = np.asarray(bm.z_where)
    gt_zp = np.asarray(bm.z_pres)

    # Frame 0 only — collect per-alive-GT-cell position + scale errors.
    pos_errs_px = []       # error in pixels (Euclidean)
    scale_errs = []        # error in scale (sigmoid space)
    x_errs_px = []         # signed x error (pixels)
    y_errs_px = []         # signed y error (pixels)

    pred_pos0 = np.tanh(pred_zw[:, 0, :, 3:5])                                             # (B, N, 2)
    pred_scale0 = 1.0 / (1.0 + np.exp(-pred_zw[:, 0, :, 0:2]))                             # (B, N, 2)
    gt_pos0 = np.tanh(gt_zw[:, 0, :, 3:5])
    gt_scale0 = 1.0 / (1.0 + np.exp(-gt_zw[:, 0, :, 0:2]))

    for v in range(args.n_videos):
        alive_pred = pred_zp[v, 0] > 0.5
        alive_gt = gt_zp[v, 0] > 0.5
        pp = pred_pos0[v][alive_pred]                                                       # (P, 2)
        gp = gt_pos0[v][alive_gt]                                                           # (G, 2)
        if len(pp) == 0 or len(gp) == 0:
            continue
        ps = pred_scale0[v][alive_pred]
        gs = gt_scale0[v][alive_gt]

        # Hungarian on position (cost = squared distance).
        cost = ((pp[:, None] - gp[None, :]) ** 2).sum(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            diff = pp[r] - gp[c]
            pos_err = np.linalg.norm(diff) * (W - 1) / 2                                    # convert to px
            pos_errs_px.append(pos_err)
            x_errs_px.append(diff[0] * (W - 1) / 2)
            y_errs_px.append(diff[1] * (H - 1) / 2)
            scale_err = np.linalg.norm(ps[r] - gs[c])
            scale_errs.append(scale_err)

    pos_errs_px = np.array(pos_errs_px)
    scale_errs = np.array(scale_errs)
    x_errs_px = np.array(x_errs_px)
    y_errs_px = np.array(y_errs_px)

    grid_res = W // (2 ** sum(1 for s in cfg.stem_strides if s == 2))
    grid_px = W / grid_res

    print(f"=== Position/scale error diagnostic ({args.n_videos} videos, frame 0) ===")
    print(f"Image size: {H}x{W}, feature grid: {grid_res}x{grid_res}, grid px: {grid_px:.1f}")
    print(f"Matched pairs: {len(pos_errs_px)}")
    print()
    print(f"Position error (pixels):")
    print(f"  mean: {pos_errs_px.mean():.2f}   median: {np.median(pos_errs_px):.2f}   "
          f"std: {pos_errs_px.std():.2f}   max: {pos_errs_px.max():.2f}")
    print(f"  fraction < 1 px:  {(pos_errs_px < 1).mean():.2%}")
    print(f"  fraction < 2 px:  {(pos_errs_px < 2).mean():.2%}")
    print(f"  fraction < 3 px:  {(pos_errs_px < 3).mean():.2%}")
    print(f"  fraction < grid_px ({grid_px:.1f}): {(pos_errs_px < grid_px).mean():.2%}")
    print(f"  fraction < 2*grid_px ({2*grid_px:.1f}): {(pos_errs_px < 2*grid_px).mean():.2%}")
    print()
    print(f"Signed x error (px): mean {x_errs_px.mean():+.2f}   std {x_errs_px.std():.2f}")
    print(f"Signed y error (px): mean {y_errs_px.mean():+.2f}   std {y_errs_px.std():.2f}")
    print(f"  (nonzero mean = systematic bias)")
    print()
    print(f"Scale error (sigmoid space):")
    print(f"  mean: {scale_errs.mean():.4f}   median: {np.median(scale_errs):.4f}   "
          f"std: {scale_errs.std():.4f}")
    print(f"  mean GT scale: {gt_scale0[gt_zp[:,0] > 0.5].mean(axis=0)}")
    print(f"  mean PRED scale: {pred_scale0[pred_zp[:,0] > 0.5].mean(axis=0)}")

    # Histogram-friendly output.
    print(f"\nPosition error histogram (bin = 1 px):")
    for lo in range(0, 15):
        c = ((pos_errs_px >= lo) & (pos_errs_px < lo + 1)).sum()
        bar = "#" * int(c * 40 / max(len(pos_errs_px), 1))
        print(f"  [{lo:2d}, {lo+1:2d}): {c:4d}  {bar}")

if __name__ == "__main__":
    main()
