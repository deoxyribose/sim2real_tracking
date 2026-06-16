"""Unsupervised adaptation loop.

Loss = reconstruction MSE + β · Σ_prior KL. **No Hungarian, no supervised matching.**

Typical usage:
    cfg = AdaptConfig(ckpt='runs/pretrain_A/ckpts/step_5000.pkl', sim_kind='many_cells_B', ...)
    train_adapt(cfg)
"""

from __future__ import annotations

import dataclasses
import os
import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import optax

from sim2real.losses.losses import AdaptLossConfig, adapt_loss
from sim2real.model.model import ModelConfig, SlotVideoModel
from sim2real.priors.registry import PriorConfig
from sim2real.train.batch import SimBatcher
from sim2real.train.ckpt import load as ckpt_load
from sim2real.train.ckpt import save as ckpt_save
from sim2real.train.freeze import make_optimizer_with_freeze
from sim2real.train.log import Logger
from sim2real.train.schedule import adamw_cosine, kl_ramp


@dataclass
class AdaptConfig:
    pretrain_ckpt: str = ""                       # path to a pretrain pickle (.pkl)
    sim_kind: str = "flagella"
    sim_cfg: object = None
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    loss_cfg: AdaptLossConfig = field(default_factory=AdaptLossConfig)
    prior_cfg: PriorConfig = field(default_factory=PriorConfig)
    freeze_patterns: tuple = ("encoder.*",)
    batch_size: int = 2
    n_steps: int = 2000
    lr_peak: float = 5e-5
    warmup_steps: int = 100
    kl_ramp_steps: int = 500
    log_every: int = 25
    ckpt_every: int = 500
    run_dir: str = "runs/adapt"
    seed: int = 0


def train_adapt(cfg: AdaptConfig) -> dict:
    rng = jax.random.key(cfg.seed)
    rng, init_key, batch_key = jax.random.split(rng, 3)

    batcher = SimBatcher(cfg.sim_kind, cfg.batch_size, cfg.sim_cfg)
    jit_sample = batcher.jit_sample()
    sample = jit_sample(batch_key)

    Nm = cfg.model_cfg.n_max

    def slice_to_model(batch):
        from sim2real.types import SimSample
        return SimSample(
            video=batch.video,
            z_where=batch.z_where[:, :, :Nm],
            z_pres=batch.z_pres[:, :, :Nm],
            z_style=batch.z_style,
            masks=batch.masks[:, :, :Nm],
            z_what=None if batch.z_what is None else batch.z_what[:, :Nm],
            meta=batch.meta,
        )

    sample_m = slice_to_model(sample)
    model = SlotVideoModel(cfg=cfg.model_cfg)

    if cfg.pretrain_ckpt and os.path.exists(cfg.pretrain_ckpt):
        ck = ckpt_load(cfg.pretrain_ckpt)
        params = ck["params"]
        print(f"loaded pretrain ckpt: {cfg.pretrain_ckpt}")
    else:
        params = model.init(init_key, sample_m.video[0], init_key)
        print("no pretrain ckpt; starting from random init")

    base_opt, lr_schedule = adamw_cosine(cfg.lr_peak, cfg.n_steps, cfg.warmup_steps)
    optimizer, mask = make_optimizer_with_freeze(base_opt, params, cfg.freeze_patterns)
    opt_state = optimizer.init(params)

    logger = Logger(cfg.run_dir)
    os.makedirs(os.path.join(cfg.run_dir, "ckpts"), exist_ok=True)

    def per_video(out, smp):
        total, m = adapt_loss(out, smp, cfg.loss_cfg, cfg.prior_cfg)
        return total, m

    @jax.jit
    def train_step(p, st, b, k):
        def loss_fn(p):
            keys = jax.random.split(k, b.video.shape[0])
            outs = jax.vmap(lambda v, kk: model.apply(p, v, kk))(b.video, keys)
            totals, metrics = jax.vmap(per_video)(outs, b)
            return jnp.mean(totals), jax.tree.map(jnp.mean, metrics)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        updates, st = optimizer.update(grads, st, p)
        p = optax.apply_updates(p, updates)
        metrics["grad_norm"] = optax.global_norm(grads)
        return p, st, loss, metrics

    rng_iter = rng
    t0 = time.time()
    last_metrics = None
    for step in range(1, cfg.n_steps + 1):
        rng_iter, k_batch, k_step = jax.random.split(rng_iter, 3)
        batch = jit_sample(k_batch)
        batch_m = slice_to_model(batch)
        params, opt_state, loss, metrics = train_step(params, opt_state, batch_m, k_step)
        last_metrics = metrics

        if step % cfg.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            print(
                f"adapt step {step:6d}  loss {float(loss):.4f}  recon {float(metrics['L_recon']):.4f}  "
                f"KL {float(metrics['L_kl']):.4f}  gnorm {float(metrics['grad_norm']):.2f}  ({elapsed:.1f}s)"
            )
            for k, v in metrics.items():
                logger.scalar(f"adapt/{k}", v, step)

        if step % cfg.ckpt_every == 0 or step == cfg.n_steps:
            ckpt_save(
                os.path.join(cfg.run_dir, "ckpts", f"adapt_{step}.pkl"),
                {"params": params, "opt_state": opt_state, "step": step},
            )

    logger.close()
    return {"params": params, "metrics": last_metrics}
