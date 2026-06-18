"""Overfit one video and decompose recon into bg vs fg contribution.

Lets us confirm whether the bg branch is hoovering up all the recon signal (cells visible in
the bg-only render), or whether the slots are doing actual foreground work (cells absent from
the bg-only render).
"""

from __future__ import annotations

import argparse
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from sim2real.losses.losses import PretrainLossConfig, pretrain_loss
from sim2real.model.background import BackgroundRenderer
from sim2real.model.model import ModelConfig, SlotVideoModel
from sim2real.priors.registry import PriorConfig
from sim2real.sim.api import build_sim
from sim2real.train.schedule import adamw_cosine
from sim2real.types import SimSample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--n-max", type=int, default=48)
    ap.add_argument("--z-style-dim", type=int, default=4)
    ap.add_argument("--bg-base-res", type=int, default=4)
    ap.add_argument("--bg-channels", type=int, nargs="+", default=[8])
    ap.add_argument("--teacher-zwhere", action="store_true")
    ap.add_argument("--teacher-zpres", action="store_true")
    ap.add_argument("--lambda-mask", type=float, default=10.0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    batch_fn, _ = build_sim("many_cells")
    key = jax.random.key(0)
    batch = batch_fn(key, 1)
    b = SimSample(
        video=batch.video,
        z_where=batch.z_where[:, :, : args.n_max],
        z_pres=batch.z_pres[:, :, : args.n_max],
        z_style=batch.z_style,
        masks=batch.masks[:, :, : args.n_max],
        z_what=None, meta=batch.meta,
    )

    cfg = ModelConfig(
        n_max=args.n_max, d_model=128, n_heads=4, n_transformer_layers=2,
        z_what_dim=64, z_style_dim=args.z_style_dim, glimpse_size=16,
        stem_channels=(16, 32, 64),
        use_background=True,
        bg_base_res=args.bg_base_res, bg_channels=tuple(args.bg_channels),
        stop_grad_recon_path=True,
    )
    model = SlotVideoModel(cfg=cfg)
    params = model.init(key, b.video[0], key)
    n_params = sum(x.size for x in jax.tree.leaves(params))
    n_bg = sum(x.size for x in jax.tree.leaves(params["params"]["bg_renderer"]))
    print(f"total params: {n_params}  bg params: {n_bg}  z_style_dim: {args.z_style_dim}", flush=True)

    opt, _ = adamw_cosine(args.lr, args.steps, warmup_steps=50)
    opt_state = opt.init(params)

    t_zw = b.z_where if args.teacher_zwhere else None
    t_zp = b.z_pres if args.teacher_zpres else None

    @jax.jit
    def step_fn(p, st, k):
        def loss_fn(p):
            keys = jax.random.split(k, 1)
            if t_zw is None and t_zp is None:
                outs = jax.vmap(lambda v, kk: model.apply(p, v, kk))(b.video, keys)
            else:
                outs = jax.vmap(
                    lambda v, kk, zw, zp: model.apply(p, v, kk, teacher_zwhere=zw, teacher_zpres=zp)
                )(b.video, keys, t_zw, t_zp)
            totals, mts = jax.vmap(
                lambda o, s: pretrain_loss(o, s, PretrainLossConfig(lambda_mask=args.lambda_mask), PriorConfig())
            )(outs, b)
            return jnp.mean(totals), jax.tree.map(jnp.mean, mts)

        (loss, mts), g = jax.value_and_grad(loss_fn, has_aux=True)(p)
        upd, st = opt.update(g, st, p)
        return optax.apply_updates(p, upd), st, loss, mts

    print(f"training {args.steps} steps...", flush=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        k = jax.random.fold_in(key, step)
        params, opt_state, loss, mts = step_fn(params, opt_state, k)
        if step in (1, 100, 500, args.steps):
            print(f"step {step:4d}  loss {float(loss):.4f}  recon {float(mts['L_recon']):.4f}  "
                  f"where {float(mts['L_where']):.4f}  pres {float(mts['L_pres']):.4f}  "
                  f"mask {float(mts['L_mask']):.4f}  ({time.time()-t0:.1f}s)", flush=True)

    # Forward and decompose
    out = model.apply(params, b.video[0], key, teacher_zwhere=t_zw[0] if t_zw is not None else None,
                       teacher_zpres=t_zp[0] if t_zp is not None else None)
    bg_renderer = BackgroundRenderer(
        out_res=128, base_res=args.bg_base_res, channels=tuple(args.bg_channels),
        z_style_dim=args.z_style_dim, use_coord_conv=True,
    )
    bg_only = bg_renderer.apply({"params": params["params"]["bg_renderer"]}, out.z_style)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for col, t in enumerate([0, 3, 6, 11]):
        axes[0, col].imshow(np.asarray(b.video[0, t, ..., 0]), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"GT t={t}")
        axes[0, col].axis("off")
    axes[1, 0].imshow(np.asarray(out.composite[0, ..., 0]), cmap="gray", vmin=0, vmax=1); axes[1, 0].set_title("full recon t=0")
    axes[1, 1].imshow(np.asarray(bg_only[..., 0]), cmap="gray", vmin=0, vmax=1); axes[1, 1].set_title("bg only")
    res = np.asarray(out.composite[0, ..., 0]) - np.asarray(bg_only[..., 0])
    axes[1, 2].imshow(res, cmap="RdBu", vmin=-0.5, vmax=0.5); axes[1, 2].set_title("composite - bg (slot contribution)")
    pred_sum = np.clip(np.asarray(out.masks_pred[0]).sum(0), 0, 1)
    axes[1, 3].imshow(pred_sum, cmap="Reds", vmin=0, vmax=1); axes[1, 3].set_title("Σ pred masks")
    for ax in axes.flat: ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "decompose.png"), dpi=100)

    print(f"\nbg pixel range: [{float(bg_only.min()):.3f}, {float(bg_only.max()):.3f}]")
    print(f"slot residual MSE: {float(np.mean(res**2)):.5f}")
    print(f"bg vs GT MSE: {float(np.mean((bg_only[..., 0] - b.video[0, 0, ..., 0])**2)):.5f}")
    print(f"wrote {args.out_dir}/decompose.png")


if __name__ == "__main__":
    main()
