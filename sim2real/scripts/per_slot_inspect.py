"""Overfit one video, then dump per-slot masks for inspection.

Tells us which of (a) empty everywhere, (b) symmetry collapse, (c) stn_write bug is the cause
of the L_mask plateau. We plot each individual slot's predicted mask alongside its GT mask and
its z_where so we can see whether the slots are at the right positions with the wrong shapes,
or vice versa.
"""

from __future__ import annotations

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from sim2real.losses.losses import PretrainLossConfig, pretrain_loss
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
    ap.add_argument("--lambda-mask", type=float, default=10.0)
    ap.add_argument("--out-dir", default="runs/per_slot_inspect")
    ap.add_argument("--use-bg", action="store_true", default=True)
    ap.add_argument("--no-bg", dest="use_bg", action="store_false")
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
        z_what_dim=64, z_style_dim=16, glimpse_size=16,
        stem_channels=(16, 32, 64),
        use_background=args.use_bg, stop_grad_recon_path=True,
    )
    model = SlotVideoModel(cfg=cfg)
    params = model.init(key, b.video[0], key)
    opt, _ = adamw_cosine(args.lr, args.steps, warmup_steps=50)
    opt_state = opt.init(params)

    @jax.jit
    def step_fn(p, st, k):
        def loss_fn(p):
            keys = jax.random.split(k, 1)
            outs = jax.vmap(
                lambda v, kk, zw, zp: model.apply(p, v, kk, teacher_zwhere=zw, teacher_zpres=zp)
            )(b.video, keys, b.z_where, b.z_pres)
            totals, mts = jax.vmap(
                lambda o, s: pretrain_loss(o, s, PretrainLossConfig(lambda_mask=args.lambda_mask), PriorConfig())
            )(outs, b)
            return jnp.mean(totals), jax.tree.map(jnp.mean, mts)

        (loss, mts), g = jax.value_and_grad(loss_fn, has_aux=True)(p)
        upd, st = opt.update(g, st, p)
        p = optax.apply_updates(p, upd)
        return p, st, loss, mts

    print(f"overfit {args.steps} steps, use_bg={args.use_bg}", flush=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        k = jax.random.fold_in(key, step)
        params, opt_state, loss, mts = step_fn(params, opt_state, k)
        if step in (1, 100, 500, args.steps):
            print(
                f"step {step:4d}  loss {float(loss):.4f}  recon {float(mts['L_recon']):.4f}  "
                f"mask {float(mts['L_mask']):.4f}  ({time.time()-t0:.1f}s)",
                flush=True,
            )

    # Forward one final time, no jit so we have full ModelOut.
    out = model.apply(params, b.video[0], key, teacher_zwhere=b.z_where[0], teacher_zpres=b.z_pres[0])

    # Per-slot dump at frame 0.
    pred_masks = np.asarray(out.masks_pred[0])                                       # (N, H, W)
    gt_masks = np.asarray(b.masks[0, 0])                                             # (N, H, W)
    gt_pres = np.asarray(b.z_pres[0, 0])                                             # (N,)
    pred_zwhere = np.asarray(out.z_where[0])                                         # (N, 3)
    gt_zwhere = np.asarray(b.z_where[0, 0])                                          # (N, 3)
    pred_zpres = np.asarray(out.z_pres[0])                                           # (N,)

    print("\nPer-slot summary (frame 0):")
    print(f"  alive GT slots (z_pres==1): {int(gt_pres.sum())}/{args.n_max}")
    print(f"  pred mask range: [{pred_masks.min():.4f}, {pred_masks.max():.4f}]")
    print(f"  pred mask mean per slot (alive only): {pred_masks[gt_pres>0.5].mean():.4f}")
    print(f"  pred mask sum per slot (alive): {pred_masks[gt_pres>0.5].sum(axis=(-1,-2))}")
    print(f"  gt mask sum per slot (alive): {gt_masks[gt_pres>0.5].sum(axis=(-1,-2))}")
    z_where_diff = np.abs(pred_zwhere - gt_zwhere).max(axis=-1)
    print(f"  max |pred z_where - gt z_where| per alive slot: max={z_where_diff[gt_pres>0.5].max():.4f}")

    _plot_slot_grid(pred_masks, gt_masks, gt_pres, gt_zwhere, args.out_dir)


def _plot_slot_grid(pred_masks, gt_masks, gt_pres, gt_zwhere, out_dir):
    """8×6 grid of 48 slots: each cell shows pred mask overlaid with the GT mask outline."""
    import matplotlib.pyplot as plt

    n = pred_masks.shape[0]
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
    for i in range(rows * cols):
        ax = axes.flat[i]
        if i >= n:
            ax.axis("off"); continue
        ax.imshow(pred_masks[i], cmap="Reds", vmin=0, vmax=1)
        # GT mask outline
        ax.contour(gt_masks[i], levels=[0.5], colors=["lime"], linewidths=0.6)
        # Position dot
        tx, ty = np.tanh(gt_zwhere[i, 1]), np.tanh(gt_zwhere[i, 2])
        h, w = pred_masks.shape[-2:]
        cx = (tx + 1.0) * 0.5 * (w - 1)
        cy = (ty + 1.0) * 0.5 * (h - 1)
        marker = "x" if gt_pres[i] < 0.5 else "+"
        color = "cyan" if gt_pres[i] >= 0.5 else "gray"
        ax.plot(cx, cy, marker=marker, color=color, markersize=8, markeredgewidth=1.5)
        title = f"#{i} {'A' if gt_pres[i] >= 0.5 else 'D'}  s={pred_masks[i].sum():.1f}"
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    fig.suptitle(
        f"Per-slot pred mask (red), GT outline (green), GT pos (cyan +); D=dead slot",
        fontsize=10,
    )
    fig.tight_layout()
    out = os.path.join(out_dir, "per_slot.png")
    plt.savefig(out, dpi=100)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
