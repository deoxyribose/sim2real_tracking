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
    ap.add_argument("--sim", default="many_cells",
                    choices=["flagella", "many_cells", "multiscale", "worms"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--n-max", type=int, default=48)
    ap.add_argument("--lambda-mask", type=float, default=10.0)
    ap.add_argument("--out-dir", default="runs/per_slot_inspect")
    ap.add_argument("--use-bg", action="store_true", default=True)
    ap.add_argument("--no-bg", dest="use_bg", action="store_false")
    ap.add_argument("--teacher-zwhere", action="store_true", default=True)
    ap.add_argument("--no-teacher-zwhere", dest="teacher_zwhere", action="store_false")
    ap.add_argument("--teacher-zpres", action="store_true", default=True)
    ap.add_argument("--no-teacher-zpres", dest="teacher_zpres", action="store_false")
    ap.add_argument("--n-groups", type=int, default=1)
    ap.add_argument("--lambda-group", type=float, default=0.0)
    ap.add_argument("--lambda-group-temp", type=float, default=0.0)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--focal-gamma", type=float, default=0.0)
    ap.add_argument("--focal-alpha", type=float, default=0.5)
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
        z_what_dim=64, z_style_dim=16, glimpse_size=16,
        stem_channels=(16, 32, 64),
        use_background=args.use_bg, stop_grad_recon_path=True,
        n_groups=args.n_groups,
    )
    model = SlotVideoModel(cfg=cfg)
    params = model.init(key, b.video[0], key)
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
            loss_cfg = PretrainLossConfig(
                lambda_mask=args.lambda_mask,
                lambda_group=args.lambda_group,
                lambda_group_temp=args.lambda_group_temp,
                dice_weight=args.dice_weight,
                focal_gamma=args.focal_gamma,
                focal_alpha=args.focal_alpha,
            )
            totals, mts = jax.vmap(
                lambda o, s: pretrain_loss(o, s, loss_cfg, PriorConfig())
            )(outs, b)
            return jnp.mean(totals), jax.tree.map(jnp.mean, mts)

        (loss, mts), g = jax.value_and_grad(loss_fn, has_aux=True)(p)
        upd, st = opt.update(g, st, p)
        p = optax.apply_updates(p, upd)
        return p, st, loss, mts

    print(f"overfit {args.sim} {args.steps} steps, use_bg={args.use_bg}, "
          f"teacher_zw={args.teacher_zwhere}, teacher_zp={args.teacher_zpres}", flush=True)
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

    out = model.apply(params, b.video[0], key,
                       teacher_zwhere=b.z_where[0] if t_zw is not None else None,
                       teacher_zpres=b.z_pres[0] if t_zp is not None else None)

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
    print(f"  z_where dim: {pred_zwhere.shape[-1]}")

    # Group posterior diagnostic.
    g_post = np.asarray(out.aux["g_post"][0])                                            # (N, K)
    if g_post.shape[-1] > 1:
        alive_idx = np.where(gt_pres > 0.5)[0]
        print(f"  n_groups: K={g_post.shape[-1]}")
        print("  per-alive-slot group posterior (argmax + max prob):")
        for n in alive_idx:
            k_star = int(np.argmax(g_post[n]))
            print(f"    slot {n:2d}: k*={k_star}  p(k*)={g_post[n, k_star]:.3f}  full={np.round(g_post[n], 2)}")
        _plot_group_attribution(g_post, gt_pres, gt_zwhere, args.out_dir)
        # All-frames group-colored composite
        all_g_post = np.asarray(out.aux["g_post"])                                       # (T, N, K)
        all_masks = np.asarray(out.masks_pred)                                           # (T, N, H, W)
        gt_video = np.asarray(b.video[0])                                                # (T, H, W, 1)
        full_composite = np.asarray(out.composite)                                       # (T, H, W, 1)
        _plot_group_colored_recon(gt_video, full_composite, all_masks, all_g_post, args.out_dir)

    _plot_slot_grid(pred_masks, gt_masks, gt_pres, gt_zwhere, args.out_dir)


_GROUP_COLORS = np.array(
    [
        [1.0, 0.20, 0.20],   # red
        [0.20, 0.60, 1.00],  # blue
        [0.20, 0.90, 0.30],  # green
        [1.00, 0.85, 0.10],  # yellow
        [0.80, 0.30, 0.95],  # purple
        [0.95, 0.55, 0.10],  # orange
        [0.10, 0.85, 0.85],  # cyan
        [0.95, 0.45, 0.75],  # pink
    ]
)


def _plot_group_colored_recon(gt_video, composite, masks, g_post, out_dir, alpha=0.6):
    """For each frame, paint each slot's mask with its argmax-group color and overlay on the GT.

    gt_video:  (T, H, W, 1)
    composite: (T, H, W, 1)
    masks:     (T, N, H, W)  — per-slot predicted masks
    g_post:    (T, N, K)     — per-slot group posterior
    """
    import matplotlib.pyplot as plt

    T = gt_video.shape[0]
    cols = min(T, 6)
    fig, axes = plt.subplots(3, cols, figsize=(2.6 * cols, 8))
    K = g_post.shape[-1]
    colors = _GROUP_COLORS[:K]

    for col, t in enumerate(np.linspace(0, T - 1, cols).astype(int)):
        # Row 0: GT
        axes[0, col].imshow(gt_video[t, ..., 0], cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"GT t={t}")
        axes[0, col].axis("off")
        # Row 1: full recon
        axes[1, col].imshow(composite[t, ..., 0], cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title("recon")
        axes[1, col].axis("off")
        # Row 2: group-colored slot masks overlaid on GT
        gt_gray = gt_video[t, ..., 0]
        # Per-slot dominant group color
        slot_groups = np.argmax(g_post[t], axis=-1)                                      # (N,)
        slot_color = colors[slot_groups]                                                 # (N, 3)
        # Soft-color overlay: pixel_color = Σ_n masks[t,n] * slot_color[n] / max(1, Σ_n masks[t,n])
        weighted = (masks[t][..., None] * slot_color[:, None, None, :]).sum(axis=0)      # (H, W, 3)
        weight_norm = np.clip(masks[t].sum(axis=0)[..., None], 0.0, 1.0)                 # (H, W, 1)
        bg = np.broadcast_to(gt_gray[..., None], (gt_gray.shape[0], gt_gray.shape[1], 3))
        rgb = (1.0 - alpha * weight_norm) * bg + alpha * weighted
        rgb = np.clip(rgb, 0.0, 1.0)
        axes[2, col].imshow(rgb)
        axes[2, col].set_title("masks colored by group")
        axes[2, col].axis("off")

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in range(K)
    ]
    labels = [f"group {k}" for k in range(K)]
    fig.legend(handles, labels, loc="upper right", ncol=K, fontsize=9, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(out_dir, "group_colored_recon.png")
    plt.savefig(out, dpi=100)
    plt.close(fig)
    print(f"wrote {out}")


def _plot_group_attribution(g_post, gt_pres, gt_zwhere, out_dir):
    """Heatmap of per-slot group posterior, with alive-slot rows highlighted."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(min(10, g_post.shape[1] * 1.2), max(4, g_post.shape[0] * 0.25)))
    im = ax.imshow(g_post, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("group k")
    ax.set_ylabel("slot n")
    ax.set_title("Group posterior q(g_n=k)  (alive slots: cyan marker)")
    for n in range(g_post.shape[0]):
        if gt_pres[n] > 0.5:
            ax.plot(-0.5, n, marker=">", color="cyan", markersize=6)
    plt.colorbar(im, ax=ax, label="prob")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "group_attribution.png"), dpi=100)
    plt.close(fig)
    print(f"wrote {out_dir}/group_attribution.png")


def _plot_slot_grid(pred_masks, gt_masks, gt_pres, gt_zwhere, out_dir):
    """Grid of N slots: pred mask (red) + GT contour (green) + GT pos (cyan)."""
    import matplotlib.pyplot as plt

    n = pred_masks.shape[0]
    cols = min(8, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8), squeeze=False)
    for i in range(rows * cols):
        ax = axes.flat[i]
        if i >= n:
            ax.axis("off"); continue
        ax.imshow(pred_masks[i], cmap="Reds", vmin=0, vmax=1)
        ax.contour(gt_masks[i], levels=[0.5], colors=["lime"], linewidths=0.6)
        # Position dot (z_where last 2 dims are tx_raw, ty_raw)
        tx, ty = np.tanh(gt_zwhere[i, -2]), np.tanh(gt_zwhere[i, -1])
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
