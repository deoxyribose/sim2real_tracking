"""Measure whether the NEM E+M iterations are converging.

Loads a ckpt, runs the model on a batch of videos, then unrolls the refiner manually for
MORE iterations than trained (default 10) and records per-iteration state. Reports:

  - ||z_where^(i+1) - z_where^(i)||   — pose displacement per iteration
  - max |z_pres^(i+1) - z_pres^(i)|   — presence change per iteration
  - responsibility entropy per slot   — attention sharpness (low entropy = sharp)
  - responsibility KL between iters   — attention change per iteration

If curves plateau after 2-3 iterations: converged (more iters won't help).
If curves keep changing at N: not converged (more iters would help).
If curves oscillate: unstable (grad explosion risk when training with more iters).

Usage:
    PYTHONPATH=. python3 -m sim2real.scripts.em_convergence \
        --ckpt runs/nem_fixed_100k/ckpts/step_100000.pkl \
        --sim many_cells_fast --n-max 18 \
        --glimpse-size 16 --d-model 128 --n-transformer-layers 3 --stem-strides 2 2 1 \
        --use-neural-em --n-iters-diagnostic 10
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

from sim2real.model.encoder import FrameEncoder
from sim2real.model.model import ModelConfig, SlotVideoModel
from sim2real.model.neural_em import NeuralEMRefiner, _normalized_grid
from sim2real.model.posenc import sinusoidal_2d
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
    ap.add_argument("--n-videos", type=int, default=16)
    ap.add_argument("--n-iters-diagnostic", type=int, default=10,
                    help="How many iterations to unroll at test time (regardless of trained N).")
    ap.add_argument("--seed", type=int, default=0)
    # Model cfg (must match ckpt).
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-transformer-layers", type=int, default=3)
    ap.add_argument("--glimpse-size", type=int, default=16)
    ap.add_argument("--stem-strides", type=int, nargs="+", default=[2, 2, 1])
    ap.add_argument("--use-neural-em", action="store_true")
    ap.add_argument("--anchor-init-fixed", action="store_true")
    ap.add_argument("--z-what-init-std", type=float, default=0.2)
    args = ap.parse_args()

    assert args.use_neural_em, "diagnostic only implemented for NEM"

    ck = ckpt_load(args.ckpt)
    params = ck["params"]
    cfg = ModelConfig(
        n_max=args.n_max, d_model=args.d_model, n_heads=4,
        n_transformer_layers=args.n_transformer_layers,
        z_what_dim=64, z_style_dim=4, glimpse_size=args.glimpse_size,
        stem_channels=(16, 32, 64), stem_strides=tuple(args.stem_strides),
        n_groups=1, use_background=True, bg_base_res=4, bg_channels=(8,),
        use_neural_em=True,
        anchor_init_fixed=args.anchor_init_fixed,
        z_what_init_std=args.z_what_init_std,
    )
    model = SlotVideoModel(cfg=cfg)

    # Sample fresh videos.
    batch_fn, _ = build_sim(args.sim)
    key = jax.random.key(args.seed)
    batch = jax.jit(lambda k: batch_fn(k, args.n_videos))(jax.random.split(key)[0])
    bm = slice_to_model(batch, args.n_max)
    print(f"loaded {args.ckpt}, {args.n_videos} videos, shape {bm.video.shape}")
    print(f"trained n_iters = {cfg.n_transformer_layers}, diagnostic unrolls to "
          f"{args.n_iters_diagnostic}")

    # Extract encoder + neural_em subtree from params.
    encoder = FrameEncoder(
        d_model=cfg.d_model, n_vit_layers=cfg.n_vit_layers,
        stem_channels=tuple(cfg.stem_channels), stem_strides=tuple(cfg.stem_strides),
    )
    enc_params = {"params": params["params"]["encoder"]}

    # Encode video[0] frame 0 across the batch — features (B, h, w, d).
    frames0 = bm.video[:, 0]                                                               # (B, H, W, C)
    feats0 = jax.jit(lambda v: jax.vmap(lambda x: encoder.apply(enc_params, x))(v))(frames0)
    feats0 = feats0[0]                                                                     # returns tuple (grid, pool)
    print("encoder output:", feats0.shape)

    # Build initial slot state (frame 0, per-video, matching model.__call__ logic).
    import math
    if cfg.anchor_init_fixed:
        side = int(math.ceil(math.sqrt(cfg.n_max)))
        lin = jnp.linspace(-0.7, 0.7, side)
        gy, gx = jnp.meshgrid(lin, lin, indexing="ij")
        pts = jnp.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)[:cfg.n_max]
        pos_raw = jnp.arctanh(jnp.clip(pts, -0.98, 0.98))
        scale_raw = jnp.full((cfg.n_max, 2), -2.2)
        theta_raw = jnp.zeros((cfg.n_max, 1))
        z_where_init = jnp.concatenate([scale_raw, theta_raw, pos_raw], axis=-1)
    else:
        z_where_init = params["params"]["z_where_init"]
    z_what_init = params["params"]["z_what_init"]                                          # (N, Zw)
    z_pres_init = jnp.full((cfg.n_max,), 0.5)

    # Get the refiner + preprocessing weights from the neural_em subtree.
    ne_params = params["params"]["neural_em"]
    # feat_norm, k_proj, v_proj live in neural_em (compact module).
    # refiner sub-module lives under neural_em/refiner.
    refiner_params_all = {"params": ne_params}   # includes feat_norm/k_proj/v_proj and refiner/*

    # We'll build K, V once per video, then call refiner in a loop.
    from sim2real.model.neural_em import NeuralEMStack
    stack = NeuralEMStack(
        n_max=cfg.n_max, d_model=cfg.d_model, z_what_dim=64,
        n_iters=args.n_iters_diagnostic, d_pos=32,
    )

    # Repurpose NeuralEMStack.apply — but we want per-iter trajectory. So write a manual loop
    # that mimics its __call__ but records after every iteration.
    def encode_KV(feat_grid):
        # Mimic NeuralEMStack preprocess.
        import flax.linen as nn
        pe = sinusoidal_2d(feat_grid.shape[0], feat_grid.shape[1], feat_grid.shape[2])
        feats_flat = (feat_grid + pe).reshape(-1, feat_grid.shape[2])
        # feat_norm, k_proj, v_proj weights are in ne_params (top level of neural_em subtree).
        ln_scale = ne_params["feat_norm"]["scale"]
        ln_bias = ne_params["feat_norm"]["bias"]
        # Manual LayerNorm.
        mu = feats_flat.mean(-1, keepdims=True)
        var = feats_flat.var(-1, keepdims=True)
        feats_flat_n = (feats_flat - mu) / jnp.sqrt(var + 1e-6) * ln_scale + ln_bias
        K = feats_flat_n @ ne_params["k_proj"]["kernel"]
        V = feats_flat_n @ ne_params["v_proj"]["kernel"]
        return K, V

    refiner = NeuralEMRefiner(d_model=cfg.d_model, d_pos=32)
    pixel_pos = _normalized_grid(feats0.shape[1], feats0.shape[2])                         # (L, 2)

    # Manual iteration loop per video, recording trajectory.
    def one_video_traj(feat_grid):
        K, V = encode_KV(feat_grid)
        z_where = z_where_init
        z_pres = z_pres_init
        z_what = z_what_init
        prev_z_what = z_what
        traj_zwhere, traj_zpres, traj_zwhat = [z_where], [z_pres], [z_what]
        for _ in range(args.n_iters_diagnostic):
            z_where, z_pres, z_what = refiner.apply(
                {"params": ne_params["refiner"]},
                K, V, pixel_pos, z_where, z_pres, z_what, prev_z_what,
            )
            traj_zwhere.append(z_where)
            traj_zpres.append(z_pres)
            traj_zwhat.append(z_what)
        return (jnp.stack(traj_zwhere), jnp.stack(traj_zpres), jnp.stack(traj_zwhat))

    trajs = jax.jit(jax.vmap(one_video_traj))(feats0)                                      # each (B, I+1, ...)
    zw_traj, zp_traj, zwhat_traj = trajs                                                   # (B, I+1, N, ...)
    zw_traj = np.asarray(zw_traj)                                                          # (B, I+1, N, 5)
    zp_traj = np.asarray(zp_traj)                                                          # (B, I+1, N)
    zwhat_traj = np.asarray(zwhat_traj)                                                    # (B, I+1, N, Zw)

    # Compute per-iteration deltas (mean over batch, slots).
    print("\n=== Per-iteration convergence (mean over batch × slots) ===")
    print("iter -> ‖Δpos‖ (in tanh-space)  |Δzpres|  ‖Δz_what‖")
    B, I1, N, _ = zw_traj.shape
    pos_traj = np.tanh(zw_traj[:, :, :, 3:5])                                              # (B, I+1, N, 2)
    for i in range(I1 - 1):
        d_pos = np.linalg.norm(pos_traj[:, i + 1] - pos_traj[:, i], axis=-1)               # (B, N)
        d_pres = np.abs(zp_traj[:, i + 1] - zp_traj[:, i])                                 # (B, N)
        d_what = np.linalg.norm(zwhat_traj[:, i + 1] - zwhat_traj[:, i], axis=-1)          # (B, N)
        print(f"  {i:2d}->{i+1:2d}    {d_pos.mean():.4f}                {d_pres.mean():.4f}   {d_what.mean():.4f}")

    # Same, but only for alive slots (z_pres > 0.5).
    print("\n=== Per-iteration convergence (ALIVE SLOTS ONLY) ===")
    for i in range(I1 - 1):
        alive_mask = zp_traj[:, i + 1] > 0.5
        d_pos = np.linalg.norm(pos_traj[:, i + 1] - pos_traj[:, i], axis=-1)
        d_pres = np.abs(zp_traj[:, i + 1] - zp_traj[:, i])
        d_what = np.linalg.norm(zwhat_traj[:, i + 1] - zwhat_traj[:, i], axis=-1)
        n_alive = alive_mask.sum()
        if n_alive > 0:
            print(f"  {i:2d}->{i+1:2d}    {d_pos[alive_mask].mean():.4f}                "
                  f"{d_pres[alive_mask].mean():.4f}   {d_what[alive_mask].mean():.4f}"
                  f"   (n_alive_iter_end = {n_alive})")

    # Save.
    out_dir = os.path.dirname(args.ckpt)
    np.savez(os.path.join(out_dir, "em_convergence.npz"),
             zw_traj=zw_traj, zp_traj=zp_traj, zwhat_traj=zwhat_traj)
    print(f"\nsaved trajectory to {os.path.join(out_dir, 'em_convergence.npz')}")


if __name__ == "__main__":
    main()
