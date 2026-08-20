"""Overfit one fixed batch — capacity diagnostic.

Sample a single batch from the simulator once, then train on it repeatedly with the SAME
config used in the full pretrain runs. If every loss term drops to ~0 within a few thousand
steps, the arch/loss config *can* represent the task and any residual eval-IoU deficit is an
optimization / generalization / data issue. Plateaus here mean the architecture or loss is
the bottleneck.

Usage:
    PYTHONPATH=. python3 -m sim2real.scripts.overfit_one \\
        --sim two_cells --n-max 4 --batch 1 --steps 5000 \\
        --glimpse-size 24 --d-model 128 --n-transformer-layers 3 --stem-strides 2 2 1 \\
        --lambda-recon 2.0 --lambda-where 5.0 --lambda-pres 1.0 --lambda-mask 0.0 \\
        --lambda-mask-glimpse 10.0 --lambda-appear-glimpse 10.0 \\
        --lambda-slot-contrast 1.0 --slot-contrast-tau 0.1 --lambda-aux 0.5 \\
        --matching-mode per_frame --match-once-after 500 \\
        --out-dir runs/overfit/two_cells_v3
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import jax
import jax.numpy as jnp

from sim2real.losses.losses import PretrainLossConfig
from sim2real.model.model import ModelConfig, SlotVideoModel
from sim2real.priors.registry import PriorConfig
from sim2real.sim.api import build_sim
from sim2real.train.ckpt import load as ckpt_load
from sim2real.train.pretrain import train_step_factory
from sim2real.train.schedule import adamw_cosine
from sim2real.types import SimSample


def slice_to_model(batch, Nm: int):
    return SimSample(
        video=batch.video,
        z_where=batch.z_where[:, :, :Nm],
        z_pres=batch.z_pres[:, :, :Nm],
        z_style=batch.z_style,
        masks=batch.masks[:, :, :Nm],
        z_what=None if batch.z_what is None else batch.z_what[:, :Nm],
        meta=batch.meta,
    )


def main():
    ap = argparse.ArgumentParser()
    # --- data / batch ---
    ap.add_argument("--sim", default="two_cells",
                    choices=["flagella", "many_cells", "easy_cells", "two_cells", "multiscale", "worms"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--n-max", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    # --- optim ---
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    # --- model ---
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-transformer-layers", type=int, default=3)
    ap.add_argument("--glimpse-size", type=int, default=24)
    ap.add_argument("--stem-strides", type=int, nargs="+", default=[2, 2, 1])
    ap.add_argument("--pres-hidden", type=int, default=128)
    ap.add_argument("--pres-depth", type=int, default=1)
    ap.add_argument("--pres-image-attn", action="store_true")
    ap.add_argument("--slot-competing-cross", action="store_true")
    ap.add_argument("--anchor-slots", action="store_true")
    ap.add_argument("--use-slot-attention", action="store_true")
    ap.add_argument("--stop-grad-zwhat", action="store_true")
    ap.add_argument("--n-groups", type=int, default=1)
    # --- loss weights ---
    ap.add_argument("--lambda-recon", type=float, default=2.0)
    ap.add_argument("--lambda-where", type=float, default=5.0)
    ap.add_argument("--lambda-pres", type=float, default=1.0)
    ap.add_argument("--lambda-mask", type=float, default=0.0)
    ap.add_argument("--lambda-mask-glimpse", type=float, default=10.0)
    ap.add_argument("--lambda-appear-glimpse", type=float, default=10.0)
    ap.add_argument("--lambda-slot-contrast", type=float, default=1.0)
    ap.add_argument("--slot-contrast-tau", type=float, default=0.1)
    ap.add_argument("--lambda-aux", type=float, default=0.5)
    ap.add_argument("--lambda-pres-smooth", type=float, default=0.0)
    ap.add_argument("--pres-focal-gamma", type=float, default=0.0)
    ap.add_argument("--pres-focal-alpha", type=float, default=0.5)
    ap.add_argument("--match-pres-weight", type=float, default=0.0)
    ap.add_argument("--matching-mode", default="per_frame", choices=["per_frame", "once"])
    ap.add_argument("--match-once-after", type=int, default=0,
                    help="If >0, switch matching_mode to 'once' at this step (one JIT recompile).")
    ap.add_argument("--hard-pres-gate", action="store_true",
                    help="Forbid dead-pred slots from winning alive-GT columns in the matcher.")
    # --- teacher forcing / bootstrap ---
    ap.add_argument("--teacher-force-zwhere", action="store_true")
    ap.add_argument("--teacher-force-zpres", action="store_true")
    ap.add_argument("--savi-bootstrap", action="store_true")
    # --- init from checkpoint ---
    ap.add_argument("--init-ckpt", default=None,
                    help="Load params from this checkpoint instead of random init. "
                         "Model config must match the checkpoint's. Optimizer state is re-init.")
    # --- logging / io ---
    ap.add_argument("--out-dir", default="runs/overfit_one")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--viz-every", type=int, default=1000,
                    help="Snapshot a viz PNG every N steps (0 disables intermediate viz).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    default_n_max = {"flagella": 8, "many_cells": 48, "easy_cells": 8, "two_cells": 4,
                     "multiscale": 16, "worms": 12}
    n_max = args.n_max if args.n_max is not None else default_n_max[args.sim]

    # 1) Freeze ONE batch on device.
    batch_fn, sim_cfg = build_sim(args.sim)
    key = jax.random.key(args.seed)
    key, k_batch = jax.random.split(key)
    batch = jax.jit(lambda k: batch_fn(k, args.batch))(k_batch)
    batch_m = slice_to_model(batch, n_max)
    T = int(batch_m.video.shape[1])
    print(f"[overfit] frozen batch: video {batch_m.video.shape}  T={T}  "
          f"z_pres alive frac {float(batch_m.z_pres.mean()):.3f}", flush=True)

    # 2) Model + loss + optimizer built to match pretrain.py.
    model_cfg = ModelConfig(
        n_max=n_max,
        d_model=args.d_model,
        n_heads=4,
        n_transformer_layers=args.n_transformer_layers,
        z_what_dim=64,
        z_style_dim=4,
        glimpse_size=args.glimpse_size,
        stem_channels=(16, 32, 64),
        stem_strides=tuple(args.stem_strides),
        n_groups=args.n_groups,
        use_background=True,
        bg_base_res=4,
        bg_channels=(8,),
        stop_grad_recon_zwhat=args.stop_grad_zwhat,
        pres_hidden=args.pres_hidden,
        pres_depth=args.pres_depth,
        pres_image_attn=args.pres_image_attn,
        slot_competing_cross=args.slot_competing_cross,
        anchor_slots=args.anchor_slots,
        use_slot_attention=args.use_slot_attention,
    )
    loss_cfg = PretrainLossConfig(
        lambda_recon=args.lambda_recon,
        lambda_where=args.lambda_where,
        lambda_pres=args.lambda_pres,
        lambda_mask=args.lambda_mask,
        lambda_kl=0.0,
        lambda_aux=args.lambda_aux,
        lambda_mask_glimpse=args.lambda_mask_glimpse,
        lambda_appear_glimpse=args.lambda_appear_glimpse,
        lambda_slot_contrast=args.lambda_slot_contrast,
        slot_contrast_tau=args.slot_contrast_tau,
        lambda_pres_smooth=args.lambda_pres_smooth,
        pres_focal_gamma=args.pres_focal_gamma,
        pres_focal_alpha=args.pres_focal_alpha,
        match_pres_weight=args.match_pres_weight,
        matching_mode=args.matching_mode,
        hard_pres_gate=args.hard_pres_gate,
    )
    prior_cfg = PriorConfig()

    model = SlotVideoModel(cfg=model_cfg)
    key, k_init = jax.random.split(key)
    params = model.init(k_init, batch_m.video[0], k_init)
    if args.init_ckpt is not None:
        loaded = ckpt_load(args.init_ckpt)
        params = loaded["params"]
        print(f"[overfit] loaded params from {args.init_ckpt} "
              f"(orig step={loaded.get('step', '?')})", flush=True)
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"[overfit] params: {n_params}", flush=True)

    optimizer, _ = adamw_cosine(args.lr, args.steps, warmup_steps=args.warmup_steps,
                                grad_clip=args.grad_clip)
    opt_state = optimizer.init(params)
    train_step = train_step_factory(
        model, loss_cfg, prior_cfg, optimizer,
        teacher_force_zwhere=args.teacher_force_zwhere,
        teacher_force_zpres=args.teacher_force_zpres,
        savi_bootstrap=args.savi_bootstrap,
    )
    matching_switched = False

    # 3) Training loop — same batch every step.
    history = []
    t0 = time.time()
    rng = key
    for step in range(1, args.steps + 1):
        if (args.match_once_after > 0 and step == args.match_once_after
                and not matching_switched):
            new_loss_cfg = dataclasses.replace(loss_cfg, matching_mode="once")
            train_step = train_step_factory(
                model, new_loss_cfg, prior_cfg, optimizer,
                teacher_force_zwhere=args.teacher_force_zwhere,
                teacher_force_zpres=args.teacher_force_zpres,
                savi_bootstrap=args.savi_bootstrap,
            )
            matching_switched = True
            print(f"[overfit] step {step}: switched matching to 'once' (JIT recompile)",
                  flush=True)

        rng = jax.random.fold_in(rng, step)
        params, opt_state, loss, metrics = train_step(params, opt_state, batch_m, rng)

        if step % args.log_every == 0 or step == 1:
            row = {k: float(v) for k, v in metrics.items()}
            row["step"] = step
            row["loss"] = float(loss)
            row["t"] = time.time() - t0
            history.append(row)
            print(
                f"step {step:6d}  loss {float(loss):.4f}  "
                f"recon {float(metrics['L_recon']):.4f}  "
                f"where {float(metrics['L_where']):.4f}  "
                f"pres {float(metrics['L_pres']):.4f}  "
                f"mask {float(metrics.get('L_mask', 0.0)):.4f}  "
                f"appearG {float(metrics.get('L_appear_glimpse', 0.0)):.4f}  "
                f"maskG {float(metrics.get('L_mask_glimpse', 0.0)):.4f}  "
                f"contr {float(metrics.get('L_slot_contrast', 0.0)):.3f}  "
                f"gnorm {float(metrics['grad_norm']):.3f}  "
                f"({row['t']:.1f}s)",
                flush=True,
            )

        if args.viz_every > 0 and (step == 1 or step % args.viz_every == 0 or step == args.steps):
            _save_viz(model, params, batch_m, rng,
                      os.path.join(args.out_dir, f"viz_step_{step:06d}.png"),
                      teacher_zwhere=batch_m.z_where[0] if args.teacher_force_zwhere else None,
                      teacher_zpres=batch_m.z_pres[0] if args.teacher_force_zpres else None,
                      bootstrap_zwhere0=batch_m.z_where[0, 0] if args.savi_bootstrap else None)

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    _plot_history(history, os.path.join(args.out_dir, "loss_curves.png"))
    print(f"\n[overfit] wrote {args.out_dir}/", flush=True)


def _save_viz(model, params, batch_m, key, path, *,
              teacher_zwhere=None, teacher_zpres=None, bootstrap_zwhere0=None):
    """Matches viz_ckpt.py: threshold masks at 0.7 and gate by z_pres before summing.
    Without the gate the padding slots' Gaussian masks (present in sim output) leak into
    the visualization even though they don't render in the composite.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    out = model.apply(params, batch_m.video[0], key,
                      teacher_zwhere=teacher_zwhere, teacher_zpres=teacher_zpres,
                      bootstrap_zwhere0=bootstrap_zwhere0)
    T = out.composite.shape[0]
    cols = min(4, T)
    fig, axes = plt.subplots(4, cols, figsize=(3 * cols, 11))
    gt_video = np.asarray(batch_m.video[0])
    pred_composite = np.asarray(out.composite)

    THR = 0.7
    gt_masks = np.asarray(batch_m.masks[0])                                  # (T, N, H, W)
    gt_alive = np.asarray(batch_m.z_pres[0])[..., None, None]                # (T, N, 1, 1)
    gt_masks_sum = np.clip(
        ((gt_masks > THR) * gt_alive).astype(np.float32).sum(1), 0, 1
    )
    pred_masks = np.asarray(out.masks_pred)
    pred_alive = np.asarray(out.z_pres)[..., None, None]
    pred_masks_sum = np.clip(
        ((pred_masks > THR) * pred_alive).astype(np.float32).sum(1), 0, 1
    )

    for t in range(cols):
        axes[0, t].imshow(gt_video[t, ..., 0], cmap="gray", vmin=0, vmax=1)
        axes[0, t].set_title(f"GT frame {t}")
        axes[1, t].imshow(pred_composite[t, ..., 0], cmap="gray", vmin=0, vmax=1)
        axes[1, t].set_title("recon")
        axes[2, t].imshow(gt_masks_sum[t], cmap="Reds", vmin=0, vmax=1)
        axes[2, t].set_title("GT masks")
        axes[3, t].imshow(pred_masks_sum[t], cmap="Reds", vmin=0, vmax=1)
        axes[3, t].set_title("pred masks")
        for r in range(4):
            axes[r, t].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close(fig)


def _plot_history(history, path):
    import matplotlib.pyplot as plt
    import numpy as np

    steps = np.array([r["step"] for r in history])
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    keys = ["loss", "L_recon", "L_where", "L_pres", "L_mask",
            "L_appear_glimpse", "L_mask_glimpse", "L_slot_contrast", "grad_norm"]
    for k in keys:
        vals = np.array([r.get(k, 0.0) for r in history])
        if np.any(vals > 0):
            ax.plot(steps, np.maximum(vals, 1e-8), label=k)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("value (log)")
    ax.set_title("overfit-one: loss components vs step")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
