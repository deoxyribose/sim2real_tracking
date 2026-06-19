"""Zoom in on individual slots so the per-slot mask shape is actually visible.

For a trained / overfit checkpoint (or a fresh overfit-on-one-video), pick the first K alive
slots and render a (zoom × zoom) crop centered on each slot's GT z_where, with three columns:
predicted mask, GT mask, overlay (pred red on GT green). At this resolution the rectangle-vs-
Gaussian shape difference is clearly visible.
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
    ap.add_argument("--sim", default="many_cells")
    ap.add_argument("--n-max", type=int, default=48)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lambda-mask", type=float, default=10.0)
    ap.add_argument("--focal-gamma", type=float, default=0.0)
    ap.add_argument("--focal-alpha", type=float, default=0.5)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--lambda-mask-glimpse", type=float, default=0.0)
    ap.add_argument("--n-show", type=int, default=8, help="how many alive slots to zoom in on")
    ap.add_argument("--zoom-px", type=int, default=24, help="crop half-extent around each slot")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    batch_fn, _ = build_sim(args.sim)
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
        z_what_dim=64, z_style_dim=4, glimpse_size=16,
        stem_channels=(16, 32, 64),
        use_background=True, stop_grad_recon_path=True,
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
            loss_cfg = PretrainLossConfig(
                lambda_mask=args.lambda_mask,
                lambda_mask_glimpse=args.lambda_mask_glimpse,
                dice_weight=args.dice_weight,
                focal_gamma=args.focal_gamma,
                focal_alpha=args.focal_alpha,
            )
            totals, _ = jax.vmap(lambda o, s: pretrain_loss(o, s, loss_cfg, PriorConfig()))(outs, b)
            return jnp.mean(totals)
        loss, g = jax.value_and_grad(loss_fn)(p)
        upd, st = opt.update(g, st, p)
        return optax.apply_updates(p, upd), st, loss

    print(f"overfit {args.sim} {args.steps} steps  focal_gamma={args.focal_gamma}  dice_weight={args.dice_weight}", flush=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        k = jax.random.fold_in(key, step)
        params, opt_state, loss = step_fn(params, opt_state, k)
        if step in (1, 100, 500, args.steps):
            print(f"step {step:4d}  loss {float(loss):.4f}  ({time.time()-t0:.1f}s)", flush=True)

    out = model.apply(params, b.video[0], key,
                       teacher_zwhere=b.z_where[0], teacher_zpres=b.z_pres[0])
    pred_masks = np.asarray(out.masks_pred[0])     # (N, H, W)
    gt_masks = np.asarray(b.masks[0, 0])           # (N, H, W)
    gt_pres = np.asarray(b.z_pres[0, 0])
    gt_zwhere = np.asarray(b.z_where[0, 0])

    alive_idx = np.where(gt_pres > 0.5)[0][: args.n_show]
    H, W = pred_masks.shape[-2:]
    z = args.zoom_px

    import matplotlib.pyplot as plt

    n = len(alive_idx)
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.6 * n))
    if n == 1:
        axes = axes[None, :]

    for row, i in enumerate(alive_idx):
        tx = np.tanh(gt_zwhere[i, -2])
        ty = np.tanh(gt_zwhere[i, -1])
        cx = int((tx + 1.0) * 0.5 * (W - 1))
        cy = int((ty + 1.0) * 0.5 * (H - 1))
        x0, x1 = max(0, cx - z), min(W, cx + z)
        y0, y1 = max(0, cy - z), min(H, cy + z)
        pred_crop = pred_masks[i, y0:y1, x0:x1]
        gt_crop = gt_masks[i, y0:y1, x0:x1]
        axes[row, 0].imshow(pred_crop, cmap="Reds", vmin=0, vmax=1)
        axes[row, 0].set_title(f"slot #{i} pred  (Σ={pred_masks[i].sum():.1f})")
        axes[row, 1].imshow(gt_crop, cmap="Greens", vmin=0, vmax=1)
        axes[row, 1].set_title(f"slot #{i} GT  (Σ={gt_masks[i].sum():.1f})")
        # Overlay: GT (green) and pred (red) on same axis
        composite = np.zeros((y1 - y0, x1 - x0, 3))
        composite[..., 0] = pred_crop
        composite[..., 1] = gt_crop
        axes[row, 2].imshow(composite, vmin=0, vmax=1)
        axes[row, 2].set_title("overlay: red=pred, green=GT")
        for c in range(3):
            axes[row, c].axis("off")

    fig.suptitle(
        f"{args.sim} overfit, focal_gamma={args.focal_gamma}, dice_weight={args.dice_weight}",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(args.out_dir, "zoom_slots.png")
    plt.savefig(out_path, dpi=110)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
