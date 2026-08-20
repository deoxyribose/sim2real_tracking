"""Visualize how slot positions evolve across E+M iterations on a test video.

Loads a ckpt, samples ONE video (seed=1 to match viz_ckpt convention), unrolls the NEM
refiner for N iterations, and plots frame 0 with slot positions overlaid at each iteration.
Each slot gets a stable color; small dot per iteration, larger dot at final iteration.

Usage:
    PYTHONPATH=. python3 -m sim2real.scripts.viz_iter_trajectory \
        --ckpt runs/nem_anchor_widewhat_10k/ckpts/step_10000.pkl \
        --sim many_cells_fast --n-max 18 \
        --glimpse-size 16 --d-model 128 --n-transformer-layers 3 --stem-strides 2 2 1 \
        --use-neural-em --anchor-init-fixed --z-what-init-std 1.0 \
        --n-iters-diagnostic 10
"""

from __future__ import annotations

import argparse
import math
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
    ap.add_argument("--seed", type=int, default=1, help="matches viz_ckpt default")
    ap.add_argument("--n-iters-diagnostic", type=int, default=10)
    # Model cfg (must match ckpt).
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-transformer-layers", type=int, default=3)
    ap.add_argument("--glimpse-size", type=int, default=16)
    ap.add_argument("--stem-strides", type=int, nargs="+", default=[2, 2, 1])
    ap.add_argument("--use-neural-em", action="store_true")
    ap.add_argument("--anchor-init-fixed", action="store_true")
    ap.add_argument("--z-what-init-std", type=float, default=0.2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

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

    batch_fn, _ = build_sim(args.sim)
    key = jax.random.key(args.seed)
    batch = batch_fn(key, 1)                                                               # single video
    bm = slice_to_model(batch, args.n_max)
    video = bm.video[0]                                                                    # (T, H, W, C)
    H, W = video.shape[1], video.shape[2]

    # Encode frame 0.
    encoder = FrameEncoder(
        d_model=cfg.d_model, n_vit_layers=cfg.n_vit_layers,
        stem_channels=tuple(cfg.stem_channels), stem_strides=tuple(cfg.stem_strides),
    )
    enc_params = {"params": params["params"]["encoder"]}
    feat_grid, _ = encoder.apply(enc_params, video[0])                                     # (h, w, d)

    # Initial slot state (mirrors model.__call__ / em_convergence).
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
    z_what_init = params["params"]["z_what_init"]
    z_pres_init = jnp.full((cfg.n_max,), 0.5)

    ne_params = params["params"]["neural_em"]
    refiner = NeuralEMRefiner(d_model=cfg.d_model, d_pos=32)
    pixel_pos = _normalized_grid(feat_grid.shape[0], feat_grid.shape[1])

    def encode_KV(feat_grid):
        pe = sinusoidal_2d(feat_grid.shape[0], feat_grid.shape[1], feat_grid.shape[2])
        feats_flat = (feat_grid + pe).reshape(-1, feat_grid.shape[2])
        ln_scale = ne_params["feat_norm"]["scale"]
        ln_bias = ne_params["feat_norm"]["bias"]
        mu = feats_flat.mean(-1, keepdims=True)
        var = feats_flat.var(-1, keepdims=True)
        feats_flat_n = (feats_flat - mu) / jnp.sqrt(var + 1e-6) * ln_scale + ln_bias
        K = feats_flat_n @ ne_params["k_proj"]["kernel"]
        V = feats_flat_n @ ne_params["v_proj"]["kernel"]
        return K, V

    K, V = encode_KV(feat_grid)

    # Unrolled iteration on frame 0.
    z_where = z_where_init
    z_pres = z_pres_init
    z_what = z_what_init
    prev_z_what = z_what
    traj_zwhere = [z_where]
    traj_zpres = [z_pres]
    for _ in range(args.n_iters_diagnostic):
        z_where, z_pres, z_what = refiner.apply(
            {"params": ne_params["refiner"]},
            K, V, pixel_pos, z_where, z_pres, z_what, prev_z_what,
        )
        traj_zwhere.append(z_where)
        traj_zpres.append(z_pres)
    zw_traj = np.asarray(jnp.stack(traj_zwhere))                                           # (I+1, N, 5)
    zp_traj = np.asarray(jnp.stack(traj_zpres))                                            # (I+1, N)

    # Positions in [-1, 1] via tanh, then map to pixel coords.
    pos_traj = np.tanh(zw_traj[:, :, 3:5])                                                 # (I+1, N, 2)
    # x, y are in [-1, 1]; map to pixel (0, W-1) and (0, H-1) — note x_pix = (x + 1) / 2 * (W-1)
    px_traj = (pos_traj[..., 0] + 1) / 2 * (W - 1)
    py_traj = (pos_traj[..., 1] + 1) / 2 * (H - 1)

    # Plot: grid of subplots, one per iteration, plus a "trajectory overlay" panel.
    import matplotlib.pyplot as plt
    I1 = zw_traj.shape[0]
    trained_iter = cfg.n_transformer_layers
    cmap = plt.get_cmap("tab20", cfg.n_max)

    ncols = 4
    nrows = int(np.ceil((I1 + 1) / ncols))   # +1 for the trajectory overlay
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)

    frame0 = np.asarray(video[0, ..., 0])
    # Consider slot "alive" if EVER alive across trajectory (so iter 0 shows something).
    ever_alive = (zp_traj > 0.5).any(axis=0)                                               # (N,)
    for it in range(I1):
        ax = axes[it]
        ax.imshow(frame0, cmap="gray", vmin=0, vmax=1)
        alive_now = zp_traj[it] > 0.5
        for k in range(cfg.n_max):
            if not ever_alive[k]:
                continue    # skip permanently-dead slots to reduce clutter
            color = cmap(k)
            # Solid + big if alive at this iter, dimmed + smaller if not-yet-alive.
            if alive_now[k]:
                ax.scatter(px_traj[it, k], py_traj[it, k], s=40, c=[color],
                           edgecolors="k", linewidths=0.5, alpha=0.95)
                ax.annotate(str(k), (px_traj[it, k], py_traj[it, k]),
                            color="white", fontsize=7, ha="center", va="center")
            else:
                # Not-yet-alive: use hollow marker so grid anchors are visible at iter 0.
                ax.scatter(px_traj[it, k], py_traj[it, k], s=60, facecolors="none",
                           edgecolors=color, linewidths=1.2, alpha=0.9)
        star = "★" if it == trained_iter else ""
        ax.set_title(f"iter {it}  {star}")
        ax.axis("off")

    # Trajectory overlay: connect positions across iterations for each slot.
    ax = axes[I1]
    ax.imshow(frame0, cmap="gray", vmin=0, vmax=1)
    for k in range(cfg.n_max):
        if not (zp_traj[-1, k] > 0.5):
            continue
        color = cmap(k)
        ax.plot(px_traj[:, k], py_traj[:, k], color=color, linewidth=1.5, alpha=0.7)
        ax.scatter(px_traj[0, k], py_traj[0, k], s=15, c=[color], marker="o",
                   edgecolors="k", linewidths=0.5)   # start
        ax.scatter(px_traj[-1, k], py_traj[-1, k], s=50, c=[color], marker="*",
                   edgecolors="k", linewidths=0.5)   # end
    ax.set_title(f"trajectory: o=iter 0, *=iter {I1 - 1}")
    ax.axis("off")

    for j in range(I1 + 1, len(axes)):
        axes[j].axis("off")

    out_path = args.out or os.path.splitext(args.ckpt)[0] + f"_iter_traj_seed{args.seed}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
