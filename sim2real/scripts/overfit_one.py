"""Overfit one fixed batch — diagnostic.

Sample a single batch from the simulator once, then train on it repeatedly. If the model has
the capacity to express the task, each loss term should drop to ~0 within a few thousand steps.
Plateaus here mean the *architecture* is the bottleneck, not training data / training time.

Usage:
    PYTHONPATH=. python3 -m sim2real.scripts.overfit_one --sim many_cells --steps 3000
"""

from __future__ import annotations

import argparse
import os
import time

import jax
import jax.numpy as jnp
import optax

from sim2real.losses.losses import PretrainLossConfig, pretrain_loss
from sim2real.model.model import ModelConfig, SlotVideoModel
from sim2real.priors.registry import PriorConfig
from sim2real.sim.api import build_sim
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
    ap.add_argument("--sim", default="many_cells")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--n-max", type=int, default=48)
    ap.add_argument("--teacher-zwhere", action="store_true")
    ap.add_argument("--teacher-zpres", action="store_true")
    ap.add_argument("--lambda-mask", type=float, default=10.0)
    ap.add_argument("--lambda-recon", type=float, default=2.0)
    ap.add_argument("--lambda-where", type=float, default=1.0)
    ap.add_argument("--lambda-pres", type=float, default=1.0)
    ap.add_argument("--out-dir", default="runs/overfit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Sample ONE fixed batch and freeze it on device.
    batch_fn, _ = build_sim(args.sim)
    key = jax.random.key(args.seed)
    batch = batch_fn(key, args.batch)
    batch_m = slice_to_model(batch, args.n_max)
    print(f"frozen batch: video {batch_m.video.shape}, z_pres alive frac: "
          f"{float(batch_m.z_pres.mean()):.3f}", flush=True)

    # 2) Build model with same config as v4.
    cfg_m = ModelConfig(
        n_max=args.n_max,
        d_model=128, n_heads=4, n_transformer_layers=2,
        z_what_dim=64, z_style_dim=16, glimpse_size=16,
        stem_channels=(16, 32, 64),
        use_background=True,
        stop_grad_recon_path=True,
    )
    model = SlotVideoModel(cfg=cfg_m)
    params = model.init(key, batch_m.video[0], key)
    print(f"params: {sum(x.size for x in jax.tree.leaves(params))}", flush=True)

    optimizer, _ = adamw_cosine(args.lr, args.steps, warmup_steps=50)
    opt_state = optimizer.init(params)

    loss_cfg = PretrainLossConfig(
        lambda_recon=args.lambda_recon,
        lambda_where=args.lambda_where,
        lambda_pres=args.lambda_pres,
        lambda_mask=args.lambda_mask,
        lambda_kl=0.0,
    )
    prior_cfg = PriorConfig()

    t_zw = batch_m.z_where if args.teacher_zwhere else None
    t_zp = batch_m.z_pres if args.teacher_zpres else None

    def fwd_one(p, v, k, tzw, tzp):
        return model.apply(p, v, k, teacher_zwhere=tzw, teacher_zpres=tzp)

    @jax.jit
    def step_fn(p, st, k):
        def loss_fn(p):
            keys = jax.random.split(k, batch_m.video.shape[0])
            if t_zw is None and t_zp is None:
                outs = jax.vmap(lambda v, kk: fwd_one(p, v, kk, None, None))(batch_m.video, keys)
            else:
                outs = jax.vmap(
                    lambda v, kk, zw, zp: fwd_one(p, v, kk, zw, zp)
                )(batch_m.video, keys, t_zw, t_zp)
            totals, metrics = jax.vmap(lambda o, s: pretrain_loss(o, s, loss_cfg, prior_cfg))(outs, batch_m)
            return jnp.mean(totals), jax.tree.map(jnp.mean, metrics)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        updates, st = optimizer.update(grads, st, p)
        p = optax.apply_updates(p, updates)
        metrics["grad_norm"] = optax.global_norm(grads)
        return p, st, loss, metrics

    # 3) Training loop with metric logging.
    history = []
    rng = key
    t0 = time.time()
    for step in range(1, args.steps + 1):
        rng = jax.random.fold_in(rng, step)
        params, opt_state, loss, metrics = step_fn(params, opt_state, rng)
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
                f"mask {float(metrics['L_mask']):.4f}  "
                f"gnorm {float(metrics['grad_norm']):.3f}  "
                f"({row['t']:.1f}s)",
                flush=True,
            )

    # 4) Save final viz of overfit batch.
    out = model.apply(params, batch_m.video[0], rng, teacher_zwhere=t_zw[0] if t_zw is not None else None,
                       teacher_zpres=t_zp[0] if t_zp is not None else None)
    _save_viz(out, batch_m, os.path.join(args.out_dir, "viz.png"))

    # 5) Save loss history JSON + plot.
    import json
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    _plot_history(history, os.path.join(args.out_dir, "loss_curves.png"))
    print(f"\nWrote {args.out_dir}/", flush=True)


def _save_viz(out, batch_m, path):
    import matplotlib.pyplot as plt
    import numpy as np

    T = out.composite.shape[0]
    cols = min(4, T)
    fig, axes = plt.subplots(4, cols, figsize=(3 * cols, 11))
    gt_video = np.asarray(batch_m.video[0])
    pred_composite = np.asarray(out.composite)
    gt_masks_sum = np.clip(np.asarray(batch_m.masks[0]).sum(1), 0, 1)
    pred_masks_sum = np.clip(np.asarray(out.masks_pred).sum(0), 0, 1)
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
    for key in ("loss", "L_recon", "L_where", "L_pres", "L_mask", "grad_norm"):
        vals = np.array([r[key] for r in history])
        ax.plot(steps, vals, label=key)
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
