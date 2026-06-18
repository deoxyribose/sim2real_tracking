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
    ap.add_argument("--bg-base-res", type=int, default=4)
    ap.add_argument("--bg-channels", type=int, nargs="+", default=[8])
    args = ap.parse_args()

    ck = ckpt_load(args.ckpt)
    params = ck["params"]
    model_cfg = ModelConfig(
        n_max=args.n_max,
        d_model=args.d_model,
        n_heads=4,
        n_transformer_layers=args.n_transformer_layers,
        z_what_dim=64,
        z_style_dim=args.z_style_dim,
        glimpse_size=args.glimpse_size,
        stem_channels=(16, 32, 64),
        n_groups=args.n_groups,
        use_background=True,
        bg_base_res=args.bg_base_res,
        bg_channels=tuple(args.bg_channels),
    )
    model = SlotVideoModel(cfg=model_cfg)

    batch_fn, _ = build_sim(args.sim)
    key = jax.random.key(args.seed)
    batch = batch_fn(key, 1)
    batch = slice_to_model(batch, args.n_max)

    @jax.jit
    def fwd(video, k):
        return model.apply(params, video, k)

    out = fwd(batch.video[0], key)

    import matplotlib.pyplot as plt

    T = batch.video.shape[1]
    cols = min(4, T)
    fig, axes = plt.subplots(4, cols, figsize=(3 * cols, 11))

    gt_video = np.asarray(batch.video[0])               # (T, H, W, 1)
    pred_composite = np.asarray(out.composite)          # (T, H, W, 1)
    gt_masks_sum = np.clip(np.asarray(batch.masks[0]).sum(1), 0, 1)  # (T, H, W)
    pred_masks_sum = np.clip(np.asarray(out.masks_pred).sum(1), 0, 1)

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
