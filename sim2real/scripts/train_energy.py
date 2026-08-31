"""Prototype trainer for the energy-score U-Net.

Single-GPU, single-file. Sim runs on the fly (JAX-jit, cheap). No wandb; we
print stats and dump a checkpoint every `--save-every` steps.

Sanity monitor: `diversity_over_accuracy = ½E‖X−X'‖ / E‖X−y‖`. If this drops
to zero within a few hundred steps, the noise channel is being ignored.
"""
from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from sim2real.model.energy_loss import (
    EnergyLossConfig, compute_energy_loss, gt_curves_from_sim,
)
from sim2real.model.unet_energy import (
    UNetConfig, UNetEnergy, decode_curves, sample_batched_noise, unpack_pred,
)
from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip


def load_pca(path: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Load PCA and rescale basis by per-mode σ so coefficients live in unit
    variance — the model predicts standardized coefficients bounded to ±3
    which then correspond to ~99% of the natural coefficient range."""
    d = np.load(path, allow_pickle=True)
    mean = np.asarray(d["mean"])
    basis = np.asarray(d["basis"])
    per_mode_var = np.asarray(d["per_mode_var"])
    sigma = np.sqrt(per_mode_var)[:, None, None]      # (M, 1, 1)
    basis_scaled = basis * sigma
    return jnp.asarray(mean), jnp.asarray(basis_scaled)


def build_train_step(model: UNetEnergy, cfg_u: UNetConfig,
                     cfg_e: EnergyLossConfig,
                     pca_mean: jnp.ndarray, pca_basis: jnp.ndarray,
                     n_max_gt: int, optimizer):
    """Returns a jit'd train_step(state, batch, key) → (state, stats)."""

    def loss_fn(params, batch, key):
        video, gt_curves, gt_valid, gt_widths, gt_amps = batch
        B = video.shape[0]
        k_a, k_b = jax.random.split(key)
        noise_a = sample_batched_noise(k_a, B, cfg_u)
        noise_b = sample_batched_noise(k_b, B, cfg_u)
        pred_a = model.apply(params, video, noise_a, train=True)
        pred_b = model.apply(params, video, noise_b, train=True)

        curves_a = decode_curves(pred_a, cfg_u, pca_mean, pca_basis)
        curves_b = decode_curves(pred_b, cfg_u, pca_mean, pca_basis)
        f_a = unpack_pred(pred_a); f_b = unpack_pred(pred_b)
        scores_a = jax.nn.sigmoid(f_a["score"])
        scores_b = jax.nn.sigmoid(f_b["score"])
        widths_a, widths_b = f_a["width"], f_b["width"]
        amps_a, amps_b = f_a["amp"], f_b["amp"]

        def per_ex(i):
            return compute_energy_loss(
                curves_a[i], curves_b[i],
                scores_a[i], scores_b[i],
                widths_a[i], widths_b[i],
                amps_a[i], amps_b[i],
                gt_curves[i], gt_valid[i],
                gt_widths[i], gt_amps[i],
                cfg_e,
            )
        # vmap over batch
        totals, stats = jax.vmap(per_ex)(jnp.arange(B))
        stats = {k: v.mean() for k, v in stats.items()}
        return totals.mean(), stats

    @partial(jax.jit, donate_argnums=(0,))
    def train_step(state, batch, key):
        (loss, stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, batch, key)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        return TrainState(params=params, opt_state=opt_state, step=state.step + 1), \
                stats
    return train_step


class TrainState:
    def __init__(self, params, opt_state, step):
        self.params, self.opt_state, self.step = params, opt_state, step

    def tree_flatten(self):
        return (self.params, self.opt_state, self.step), None
    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


jax.tree_util.register_pytree_node_class(TrainState)


def sample_batch(key, sim_cfg: DiverseSimConfig, batch_size: int,
                 cfg_u: UNetConfig, n_max_gt: int):
    """Draw one training batch: (video, gt_curves, gt_valid, gt_widths, gt_amps).
    We pass the mid-frame gt only (deeptangle predicts past/present/future;
    we start with just the middle for simplicity)."""
    keys = jax.random.split(key, batch_size)
    outs = jax.vmap(lambda k: sample_clip(k, sim_cfg))(keys)
    clip = outs["clip_median"]                                 # (B, T, H, W)
    gt_curves = outs["curves"][:, sim_cfg.T // 2]              # (B, N, K, 2)
    gt_valid = outs["flagella"]["alive"]                       # (B, N)
    gt_widths = outs["flagella"]["width"]                      # (B, N)
    gt_amps = outs["flagella"]["amp"]                          # (B, N) signed
    return clip, gt_curves, gt_valid, gt_widths, gt_amps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca", default="data_cache/flagella_pca.npz")
    ap.add_argument("--n-steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--out-dir", default="runs/energy_v0")
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--beta", type=float, default=0.5,
                    help="weight on the diversity term (0=no spread, 1=proper ES)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Configs
    cfg_u = UNetConfig(T=args.T, H=args.H, W=args.H)
    cfg_e = EnergyLossConfig(beta=args.beta)
    sim_cfg = DiverseSimConfig(T=args.T, H=args.H, W=args.H)

    print(f"grid: {cfg_u.grid_h}x{cfg_u.grid_w}  "
          f"n_suggestions={cfg_u.n_suggestions}  n_out_per_pred={cfg_u.n_out_per_pred}")
    print(f"input: T={cfg_u.T}, {cfg_u.H}x{cfg_u.W}")

    pca_mean, pca_basis = load_pca(args.pca)
    print(f"loaded PCA: mean {pca_mean.shape}, basis {pca_basis.shape}")

    # Initialise model
    model = UNetEnergy(cfg=cfg_u)
    key = jax.random.key(args.seed)
    k_init, k_train = jax.random.split(key)
    dummy_video = jnp.zeros((1, cfg_u.T, cfg_u.H, cfg_u.W))
    dummy_noise = jnp.zeros((1, cfg_u.H, cfg_u.W, 1))
    params = model.init(k_init, dummy_video, dummy_noise, train=True)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"model params: {n_params/1e6:.2f} M")

    # Global-norm gradient clipping cures the occasional single-example gradient
    # explosion that produced NaNs in earlier runs.
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(args.lr, weight_decay=1e-4),
    )
    opt_state = optimizer.init(params)
    state = TrainState(params=params, opt_state=opt_state, step=jnp.array(0))

    n_max_gt = sim_cfg.n_max_flagella   # fixed shape for the loss
    train_step = build_train_step(model, cfg_u, cfg_e, pca_mean, pca_basis,
                                    n_max_gt, optimizer)

    # Precompile once
    print("compiling...", flush=True)
    key, k_b, k_s = jax.random.split(k_train, 3)
    batch = sample_batch(k_b, sim_cfg, args.batch_size, cfg_u, n_max_gt)
    state, stats = train_step(state, batch, k_s)
    stats = jax.device_get(stats)
    print("first-step stats:", {k: float(v) for k, v in stats.items()})

    t0 = time.time()
    for step in range(1, args.n_steps + 1):
        key, k_b, k_s = jax.random.split(key, 3)
        batch = sample_batch(k_b, sim_cfg, args.batch_size, cfg_u, n_max_gt)
        state, stats = train_step(state, batch, k_s)
        if step % args.log_every == 0:
            stats = jax.device_get(stats)
            dt = time.time() - t0
            print(f"[step {step:5d}]  t={dt:.1f}s  "
                  f"loss={float(stats['loss_total']):.3f}  "
                  f"acc={float(stats['loss_accuracy']):.3f}  "
                  f"div={float(stats['loss_diversity']):.3f}  "
                  f"div/acc={float(stats['diversity_over_accuracy']):.3f}  "
                  f"score={float(stats['loss_score']):.4f}  "
                  f"n_gt={float(stats['n_gt']):.1f}",
                  flush=True)
        if step % args.save_every == 0:
            # Simple pickle dump
            import pickle
            (out_dir / f"ckpt_step{step:06d}.pkl").write_bytes(
                pickle.dumps(dict(params=jax.device_get(state.params),
                                    step=int(state.step),
                                    cfg_u=cfg_u.__dict__)))
            print(f"  saved checkpoint at step {step}")

    print("done.")


if __name__ == "__main__":
    main()
