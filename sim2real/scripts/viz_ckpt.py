"""Render a checkpoint's predictions on a fresh batch.

Saves a 4-frame side-by-side PNG:
  row 0: GT video
  row 1: model composite (clipped recon)
  row 2: GT mask sum
  row 3: predicted mask sum

Usage:
    PYTHONPATH=. python3 -m sim2real.scripts.viz_ckpt --ckpt runs/<exp>/ckpts/step_500.pkl --sim many_cells
"""

from __future__ import annotations

import argparse
import os

import jax
import numpy as np

from sim2real.model.model import ModelConfig, SlotVideoModel
from sim2real.scripts.eval_ckpt import slice_to_model
from sim2real.sim.api import build_sim
from sim2real.train.ckpt import load as ckpt_load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sim", default="many_cells")
    ap.add_argument("--n-max", type=int, default=48)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=1)
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
    ap.add_argument("--use-isa", action="store_true")
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
    ap.add_argument("--savi-bootstrap", action="store_true",
                    help="Pass GT z_where[:, 0] as slot init per video. Required if the model "
                         "was trained with --savi-bootstrap.")
    args = ap.parse_args()

    ck = ckpt_load(args.ckpt)
    params = ck["params"]
    model_cfg = ModelConfig(
        n_max=args.n_max,
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
    )
    model = SlotVideoModel(cfg=model_cfg)

    batch_fn, _ = build_sim(args.sim)
    key = jax.random.key(args.seed)
    batch = batch_fn(key, 1)
    batch = slice_to_model(batch, args.n_max)

    boot0 = batch.z_where[0, 0] if args.savi_bootstrap else None

    @jax.jit
    def fwd(video, k, b):
        return model.apply(params, video, k, bootstrap_zwhere0=b)

    out = fwd(batch.video[0], key, boot0)

    import matplotlib.pyplot as plt

    T = batch.video.shape[1]
    cols = min(4, T)
    fig, axes = plt.subplots(4, cols, figsize=(3 * cols, 11))

    gt_video = np.asarray(batch.video[0])               # (T, H, W, 1)
    pred_composite = np.asarray(out.composite)          # (T, H, W, 1)
    # Threshold at 0.7 + gate dead slots by z_pres. The sim generates masks for ALL n_max
    # slots including padding; only alive ones actually render. Without the gate, the sum
    # includes ~2-3x extra phantom masks from dead slots (bug seen 2026-08-13).
    THR = 0.7
    gt_masks = np.asarray(batch.masks[0])                                        # (T, N, H, W)
    gt_alive = np.asarray(batch.z_pres[0])[..., None, None]                      # (T, N, 1, 1)
    gt_masks_sum = np.clip(
        ((gt_masks > THR) * gt_alive).astype(np.float32).sum(1), 0, 1
    )                                                                             # (T, H, W)
    pred_masks = np.asarray(out.masks_pred)                                       # (T, N, H, W)
    pred_alive = np.asarray(out.z_pres)[..., None, None]                          # (T, N, 1, 1)
    pred_masks_sum = np.clip(
        ((pred_masks > THR) * pred_alive).astype(np.float32).sum(1), 0, 1
    )

    for t in range(cols):
        axes[0, t].imshow(gt_video[t, ..., 0], cmap="gray", vmin=0, vmax=1)
        axes[0, t].set_title(f"GT frame {t}")
        axes[1, t].imshow(pred_composite[t, ..., 0], cmap="gray", vmin=0, vmax=1)
        axes[1, t].set_title(f"recon frame {t}")
        axes[2, t].imshow(gt_masks_sum[t], cmap="Reds", vmin=0, vmax=1)
        axes[2, t].set_title("GT mask sum")
        axes[3, t].imshow(pred_masks_sum[t], cmap="Reds", vmin=0, vmax=1)
        axes[3, t].set_title("pred mask sum")
        for r in range(4):
            axes[r, t].axis("off")

    out_path = args.out or os.path.splitext(args.ckpt)[0] + "_viz.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=90)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
