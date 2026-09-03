"""Cell-coverage diagnostic: which GT cells does the model actually claim?

For a test video, plots:
  - GT cells (red X at position, with cell mask underneath)
  - Alive pred slots (blue filled dot, labeled with slot index)
  - Dead pred slots (grey hollow dot)
  - MISSED cells (GT cell with no alive slot within `--match-radius`): highlighted red circle
  - DUPLICATE claims (2+ alive slots within `--match-radius` of same cell): connected lines

Usage:
    PYTHONPATH=. python3 -m sim2real.scripts.viz_cell_coverage \
        --ckpt runs/nem_anchor_widewhat_10k/ckpts/step_10000.pkl \
        --sim many_cells_fast --n-max 18 \
        --glimpse-size 16 --d-model 128 --n-transformer-layers 3 --stem-strides 2 2 1 \
        --use-neural-em --anchor-init-fixed --z-what-init-std 1.0 --seed 1
"""

from __future__ import annotations

import argparse
import os

import jax
import numpy as np

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
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--match-radius", type=float, default=0.08,
                    help="Distance in normalized [-1,1] coords within which a pred slot 'covers' "
                         "a GT cell. 0.08 ≈ 2.5 px in 64x64.")
    ap.add_argument("--pres-threshold", type=float, default=0.5)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-transformer-layers", type=int, default=3)
    ap.add_argument("--glimpse-size", type=int, default=16)
    ap.add_argument("--stem-strides", type=int, nargs="+", default=[2, 2, 1])
    ap.add_argument("--use-neural-em", action="store_true")
    ap.add_argument("--use-isa", action="store_true")
    ap.add_argument("--anchor-init-fixed", action="store_true")
    ap.add_argument("--z-what-init-std", type=float, default=0.2)
    ap.add_argument("--nem-attn-temp", type=float, default=1.0)
    ap.add_argument("--nem-use-bg-slot", action="store_true")
    ap.add_argument("--nem-dual-source-what", action="store_true")
    ap.add_argument("--nem-use-pca-theta", action="store_true")
    ap.add_argument("--decoder-use-film", action="store_true")
    ap.add_argument("--nem-theta-delta-scale", type=float, default=0.05)
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
        use_neural_em=args.use_neural_em, use_isa=args.use_isa,
        anchor_init_fixed=args.anchor_init_fixed, z_what_init_std=args.z_what_init_std,
        nem_attn_temp=args.nem_attn_temp,
        nem_use_bg_slot=args.nem_use_bg_slot,
        nem_dual_source_what=args.nem_dual_source_what,
        nem_use_pca_theta=args.nem_use_pca_theta,
        decoder_use_film=args.decoder_use_film,
        nem_theta_delta_scale=args.nem_theta_delta_scale,
    )
    model = SlotVideoModel(cfg=cfg)

    batch_fn, _ = build_sim(args.sim)
    key = jax.random.key(args.seed)
    batch = batch_fn(key, 1)
    bm = slice_to_model(batch, args.n_max)
    video = np.asarray(bm.video[0])                                                        # (T, H, W, C)
    gt_zwhere = np.asarray(bm.z_where[0])                                                  # (T, N, 5)
    gt_zpres = np.asarray(bm.z_pres[0])                                                    # (T, N)

    out = jax.jit(lambda v, k: model.apply(params, v, k))(bm.video[0], key)
    pred_zwhere = np.asarray(out.z_where)                                                  # (T, N, 5)
    pred_zpres = np.asarray(out.z_pres)                                                    # (T, N)

    # Focus on frame 0.
    T, H, W = video.shape[0], video.shape[1], video.shape[2]
    gt_pos0 = np.tanh(gt_zwhere[0, :, 3:5])                                                # (N, 2)
    gt_alive0 = gt_zpres[0] > 0.5
    pred_pos0 = np.tanh(pred_zwhere[0, :, 3:5])                                            # (N, 2)
    pred_alive0 = pred_zpres[0] > args.pres_threshold

    gt_alive_pos = gt_pos0[gt_alive0]                                                      # (G, 2)
    pred_alive_pos = pred_pos0[pred_alive0]                                                # (P, 2)
    pred_alive_ids = np.where(pred_alive0)[0]

    # Coverage analysis.
    # For each GT cell, find alive pred slots within match_radius.
    diffs = pred_alive_pos[None, :, :] - gt_alive_pos[:, None, :]                          # (G, P, 2)
    dists = np.linalg.norm(diffs, axis=-1)                                                 # (G, P)
    covered_by = [np.where(dists[g] < args.match_radius)[0].tolist() for g in range(len(gt_alive_pos))]
    n_missed = sum(1 for c in covered_by if len(c) == 0)
    n_duplicated = sum(1 for c in covered_by if len(c) >= 2)

    # Pred slots that fall on no GT cell within match_radius = "phantom" claims.
    if len(gt_alive_pos) > 0:
        min_pred_to_gt = dists.min(axis=0)
    else:
        min_pred_to_gt = np.full(len(pred_alive_pos), np.inf)
    phantom_pred = pred_alive_ids[min_pred_to_gt > args.match_radius]

    print(f"=== Cell coverage on seed={args.seed} frame 0 ===")
    print(f"GT alive cells: {len(gt_alive_pos)}")
    print(f"Pred alive slots: {len(pred_alive_pos)}")
    print(f"Missed GT cells: {n_missed}")
    print(f"Duplicated (>=2 alive slots on same cell): {n_duplicated}")
    print(f"Phantom pred slots (alive but not near any GT cell): {len(phantom_pred)}")
    if len(phantom_pred) > 0:
        print(f"  phantom slot IDs: {phantom_pred.tolist()}")

    # Pixel coords.
    def pos_to_px(p):
        return (p[..., 0] + 1) / 2 * (W - 1), (p[..., 1] + 1) / 2 * (H - 1)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    frame0 = video[0, ..., 0]

    # Panel 1: GT truth (cells + masks).
    ax = axes[0]
    ax.imshow(frame0, cmap="gray", vmin=0, vmax=1)
    gtx, gty = pos_to_px(gt_alive_pos)
    ax.scatter(gtx, gty, s=100, marker="X", c="red", edgecolors="white", linewidths=1.5,
               label=f"GT cells ({len(gt_alive_pos)})")
    ax.set_title(f"GT: {len(gt_alive_pos)} alive cells")
    ax.legend(loc="upper right")
    ax.axis("off")

    # Panel 2: prediction overlay.
    ax = axes[1]
    ax.imshow(frame0, cmap="gray", vmin=0, vmax=1)
    # GT cells (background reference).
    ax.scatter(gtx, gty, s=180, marker="o", facecolors="none", edgecolors="red",
               linewidths=1.5, label="GT cells")

    # Alive pred slots.
    px_a, py_a = pos_to_px(pred_alive_pos)
    ax.scatter(px_a, py_a, s=60, marker="o", c="dodgerblue", edgecolors="k", linewidths=0.5,
               label=f"pred alive (z_pres>{args.pres_threshold}, n={len(pred_alive_pos)})")
    for i, sid in enumerate(pred_alive_ids):
        ax.annotate(str(sid), (px_a[i], py_a[i]), color="white", fontsize=8,
                    ha="center", va="center", weight="bold")

    # Dead pred slots.
    pred_dead_pos = pred_pos0[~pred_alive0]
    pred_dead_ids = np.where(~pred_alive0)[0]
    px_d, py_d = pos_to_px(pred_dead_pos)
    ax.scatter(px_d, py_d, s=30, marker="x", c="grey", alpha=0.6,
               label=f"pred dead (n={len(pred_dead_pos)})")

    # Missed GT cells: red circle around them.
    for g_idx, c in enumerate(covered_by):
        if len(c) == 0:
            gxp, gyp = pos_to_px(gt_alive_pos[g_idx])
            ax.scatter(gxp, gyp, s=280, marker="o", facecolors="none",
                       edgecolors="orange", linewidths=2.5)

    # Duplicated GT cells: draw line between slot pairs claiming same cell.
    for g_idx, c in enumerate(covered_by):
        if len(c) >= 2:
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    p_i = pred_alive_pos[c[i]]
                    p_j = pred_alive_pos[c[j]]
                    xi, yi = pos_to_px(p_i)
                    xj, yj = pos_to_px(p_j)
                    ax.plot([xi, xj], [yi, yj], color="magenta", linewidth=2, alpha=0.7)

    title = (f"pred: {len(pred_alive_pos)} alive / {len(pred_dead_pos)} dead\n"
             f"missed {n_missed}, dup {n_duplicated}, phantom {len(phantom_pred)}")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")

    out_path = args.out or os.path.splitext(args.ckpt)[0] + f"_coverage_seed{args.seed}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
