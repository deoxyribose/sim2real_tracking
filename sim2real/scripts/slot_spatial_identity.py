"""Diagnose per-slot SPATIAL identity across videos.

The `matching_stability.py` peak_assign metric checks whether pred slot k reliably matches
GT slot k across videos — but Hungarian matches by position and GT slot indices are arbitrary
in our sim, so this metric can't detect a pred model that binds spatially (slot k → "cell
near anchor k") without also binding by GT slot index. This diagnostic measures spatial
identity directly:

  For each slot k, run the model on B fresh videos and record pred_z_where[:, 0, k, :] (frame-0
  slot position across videos). Report:
    - mean_pos_k, std_pos_k across videos
    - if anchor_init_fixed: distance from mean_pos_k to anchor_k
    - overall spatial-identity score: 1 - mean(std_pos_k) / expected_std_if_random

A strongly spatially-bound model has std_pos ≪ std of GT cell positions.

Usage:
    PYTHONPATH=. python3 -m sim2real.scripts.slot_spatial_identity \
        --ckpt runs/nem_anchor_widewhat_10k/ckpts/step_10000.pkl \
        --sim many_cells_fast --n-max 18 \
        --glimpse-size 16 --d-model 128 --n-transformer-layers 3 --stem-strides 2 2 1 \
        --use-neural-em --anchor-init-fixed --z-what-init-std 1.0 \
        --n-videos 128
"""

from __future__ import annotations

import argparse
import math
import os

import jax
import jax.numpy as jnp
import numpy as np

from sim2real.model.model import ModelConfig, SlotVideoModel
from sim2real.sim.api import build_sim
from sim2real.train.ckpt import load as ckpt_load
from sim2real.types import SimSample


def slice_to_model(batch, Nm):
    return SimSample(
        video=batch.video,
        z_where=batch.z_where[:, :, :Nm],
        z_pres=batch.z_pres[:, :, :Nm],
        z_style=batch.z_style,
        masks=batch.masks[:, :, :Nm],
        z_what=None if batch.z_what is None else batch.z_what[:, :Nm],
        meta=batch.meta,
    )


def _anchor_grid(n_max: int) -> np.ndarray:
    """Recompute the fixed anchor grid used when --anchor-init-fixed is set."""
    side = int(math.ceil(math.sqrt(n_max)))
    lin = np.linspace(-0.7, 0.7, side)
    gy, gx = np.meshgrid(lin, lin, indexing="ij")
    return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)[:n_max]                     # (N, 2) — (x, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--n-max", type=int, required=True)
    ap.add_argument("--n-videos", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    # Model config (must match ckpt).
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-transformer-layers", type=int, default=3)
    ap.add_argument("--glimpse-size", type=int, default=16)
    ap.add_argument("--stem-strides", type=int, nargs="+", default=[2, 2, 1])
    ap.add_argument("--use-neural-em", action="store_true")
    ap.add_argument("--use-isa", action="store_true")
    ap.add_argument("--anchor-init-fixed", action="store_true")
    ap.add_argument("--z-what-init-std", type=float, default=0.2)
    args = ap.parse_args()

    ck = ckpt_load(args.ckpt)
    params = ck["params"]
    cfg = ModelConfig(
        n_max=args.n_max, d_model=args.d_model, n_heads=4,
        n_transformer_layers=args.n_transformer_layers,
        z_what_dim=64, z_style_dim=4, glimpse_size=args.glimpse_size,
        stem_channels=(16, 32, 64), stem_strides=tuple(args.stem_strides),
        n_groups=1, use_background=True, bg_base_res=4, bg_channels=(8,),
        use_neural_em=args.use_neural_em, use_isa=args.use_isa,
        anchor_init_fixed=args.anchor_init_fixed,
        z_what_init_std=args.z_what_init_std,
    )
    model = SlotVideoModel(cfg=cfg)

    # Sample fresh videos.
    batch_fn, _ = build_sim(args.sim)
    key = jax.random.key(args.seed)
    key, kb = jax.random.split(key)
    batch = jax.jit(lambda k: batch_fn(k, args.n_videos))(kb)
    bm = slice_to_model(batch, args.n_max)
    print(f"loaded {args.ckpt}, {args.n_videos} videos, shape {bm.video.shape}", flush=True)

    # Forward in chunks.
    CHUNK = 8

    @jax.jit
    def fwd(video, k):
        keys = jax.random.split(k, video.shape[0])
        return jax.vmap(lambda v, kk: model.apply(params, v, kk))(video, keys)

    key, kf = jax.random.split(key)
    zw_chunks, zp_chunks = [], []
    for i in range(0, args.n_videos, CHUNK):
        kf, kc = jax.random.split(kf)
        out = fwd(bm.video[i:i+CHUNK], kc)
        zw_chunks.append(np.asarray(out.z_where))
        zp_chunks.append(np.asarray(out.z_pres))
    pred_zw = np.concatenate(zw_chunks, axis=0)                                            # (B, T, N, 5)
    pred_zp = np.concatenate(zp_chunks, axis=0)                                            # (B, T, N)

    # Decode positions.
    pred_pos = np.tanh(pred_zw[:, 0, :, 3:5])                                              # (B, N, 2) — frame 0
    pred_scale = 1.0 / (1.0 + np.exp(-pred_zw[:, 0, :, 0:2]))                              # (B, N, 2)

    # Per-slot stats across videos.
    mean_pos = pred_pos.mean(axis=0)                                                       # (N, 2)
    std_pos = pred_pos.std(axis=0)                                                         # (N, 2)
    mean_scale = pred_scale.mean(axis=0)                                                   # (N, 2)
    mean_pres = pred_zp[:, 0].mean(axis=0)                                                 # (N,)
    std_pos_scalar = np.linalg.norm(std_pos, axis=-1)                                      # (N,)

    # Reference: std of GT cell positions (uniformly distributed, roughly).
    gt_pos = np.tanh(np.asarray(bm.z_where)[:, 0, :, 3:5])                                 # (B, N, 2)
    gt_alive = np.asarray(bm.z_pres)[:, 0]                                                 # (B, N)
    # Restrict to alive GT.
    alive_mask = gt_alive > 0.5
    gt_alive_pos = gt_pos[alive_mask]                                                       # (K, 2)
    gt_std = gt_alive_pos.std(axis=0)                                                       # (2,)
    gt_std_scalar = float(np.linalg.norm(gt_std))

    # Spatial identity score.
    # If slots were random per video: std_pos ≈ gt_std.
    # If slots are spatially bound: std_pos ≪ gt_std.
    # Score in [0, 1]: 1 = perfect binding, 0 = random.
    spatial_id_score = float(np.clip(1.0 - std_pos_scalar.mean() / (gt_std_scalar + 1e-6), 0.0, 1.0))

    # If anchor init: distance from mean_pos to anchor.
    if args.anchor_init_fixed:
        anchors = _anchor_grid(args.n_max)                                                 # (N, 2)
        anchor_dist = np.linalg.norm(mean_pos - anchors, axis=-1)                          # (N,)
    else:
        anchors = None
        anchor_dist = None

    print()
    print(f"=== Slot spatial identity on {args.n_videos} fresh {args.sim} videos ===")
    print(f"GT cell position std (isotropic): {gt_std_scalar:.3f}  (per axis: {gt_std[0]:.3f}, {gt_std[1]:.3f})")
    print(f"Mean per-slot pos std across videos: {std_pos_scalar.mean():.3f}")
    print(f"Spatial identity score (1 = perfect, 0 = random): {spatial_id_score:.3f}")
    print()

    header = "slot  mean_pres  mean_pos(x,y)      std_pos    mean_scale(sx,sy)"
    if anchors is not None:
        header += "   anchor(x,y)     dist_to_anchor"
    print(header)
    for k in range(args.n_max):
        line = (
            f"{k:3d}   {mean_pres[k]:.3f}     "
            f"({mean_pos[k, 0]:+.3f},{mean_pos[k, 1]:+.3f})  "
            f"{std_pos_scalar[k]:.3f}      "
            f"({mean_scale[k, 0]:.3f},{mean_scale[k, 1]:.3f})"
        )
        if anchors is not None:
            line += f"   ({anchors[k, 0]:+.2f},{anchors[k, 1]:+.2f})    {anchor_dist[k]:.3f}"
        print(line)

    # Also dump a summary JSON.
    out_dir = os.path.dirname(args.ckpt)
    import json
    with open(os.path.join(out_dir, "slot_spatial_identity.json"), "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "n_videos": args.n_videos,
            "gt_pos_std": gt_std.tolist(),
            "gt_pos_std_scalar": gt_std_scalar,
            "mean_slot_pos_std": float(std_pos_scalar.mean()),
            "spatial_identity_score": spatial_id_score,
            "per_slot_mean_pos": mean_pos.tolist(),
            "per_slot_std_pos": std_pos.tolist(),
            "per_slot_mean_pres": mean_pres.tolist(),
            "per_slot_anchor": anchors.tolist() if anchors is not None else None,
            "per_slot_anchor_dist": anchor_dist.tolist() if anchor_dist is not None else None,
        }, f, indent=2)
    print(f"\nwrote {os.path.join(out_dir, 'slot_spatial_identity.json')}")


if __name__ == "__main__":
    main()
