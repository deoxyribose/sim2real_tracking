"""Supervised pretraining loop.

API:
    train_pretrain(cfg) -> dict  # returns final params + metrics

`cfg` is a dataclass-style ConfigDict with everything needed (see configs/experiment/*.py).
"""

from __future__ import annotations

import dataclasses
import functools
import os
import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import optax

from sim2real.losses.losses import PretrainLossConfig, pretrain_loss
from sim2real.model.model import ModelConfig, SlotVideoModel
from sim2real.priors.registry import PriorConfig
from sim2real.train.batch import SimBatcher
from sim2real.train.ckpt import save as ckpt_save
from sim2real.train.log import Logger
from sim2real.train.schedule import adamw_cosine


@dataclass
class PretrainConfig:
    sim_kind: str = "flagella"
    sim_cfg: object = None                         # if None, uses default for sim_kind
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    loss_cfg: PretrainLossConfig = field(default_factory=PretrainLossConfig)
    prior_cfg: PriorConfig = field(default_factory=PriorConfig)
    batch_size: int = 2
    n_steps: int = 2000
    lr_peak: float = 1e-4
    warmup_steps: int = 200
    grad_clip: float = 1.0
    log_every: int = 25
    ckpt_every: int = 500
    run_dir: str = "runs/pretrain"
    seed: int = 0


def train_step_factory(model, loss_cfg, prior_cfg, optimizer):
    """Build a jitted train_step closure."""

    def model_forward_one(params, video, key):
        return model.apply(params, video, key)

    def loss_fn(params, batch, key):
        keys = jax.random.split(key, batch.video.shape[0])
        outs = jax.vmap(lambda v, k: model_forward_one(params, v, k))(batch.video, keys)

        def per_video(out, smp):
            total, metrics = pretrain_loss(out, smp, loss_cfg, prior_cfg)
            return total, metrics

        totals, metrics = jax.vmap(per_video)(outs, batch)
        loss = jnp.mean(totals)
        return loss, jax.tree.map(jnp.mean, metrics)

    @jax.jit
    def train_step(params, opt_state, batch, key):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics["grad_norm"] = optax.global_norm(grads)
        return params, opt_state, loss, metrics

    return train_step


def train_pretrain(cfg: PretrainConfig) -> dict:
    rng = jax.random.key(cfg.seed)
    rng, init_key, batch_key = jax.random.split(rng, 3)

    # Sim
    batcher = SimBatcher(cfg.sim_kind, cfg.batch_size, cfg.sim_cfg)
    jit_sample = batcher.jit_sample()
    sample = jit_sample(batch_key)

    # Model: pad / trim sample.z_pres / z_where / masks to model n_max if necessary.
    # For simplicity we assume sim_cfg.common.n_max >= model_cfg.n_max and slice on the fly.
    Nm = cfg.model_cfg.n_max
    Ns = sample.z_where.shape[2]
    if Ns < Nm:
        raise ValueError(f"sim n_max={Ns} < model n_max={Nm}; reduce model n_max or use a sim cfg with larger n_max")

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
    params = model.init(init_key, sample_m.video[0], init_key)
    print(f"params: {sum(x.size for x in jax.tree.leaves(params))}")

    optimizer, lr_schedule = adamw_cosine(
        cfg.lr_peak, cfg.n_steps, cfg.warmup_steps, grad_clip=cfg.grad_clip
    )
    opt_state = optimizer.init(params)
    train_step = train_step_factory(model, cfg.loss_cfg, cfg.prior_cfg, optimizer)

    logger = Logger(cfg.run_dir)
    os.makedirs(os.path.join(cfg.run_dir, "ckpts"), exist_ok=True)

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
                f"step {step:6d}  loss {float(loss):.4f}  recon {float(metrics['L_recon']):.4f}  "
                f"where {float(metrics['L_where']):.4f}  pres {float(metrics['L_pres']):.4f}  "
                f"mask {float(metrics['L_mask']):.4f}  "
                f"gnorm {float(metrics['grad_norm']):.2f}  "
                f"({elapsed:.1f}s)",
                flush=True,
            )
            for k, v in metrics.items():
                logger.scalar(f"train/{k}", v, step)
            logger.scalar("train/lr", float(lr_schedule(step)), step)

        if step % cfg.ckpt_every == 0 or step == cfg.n_steps:
            ckpt_save(
                os.path.join(cfg.run_dir, "ckpts", f"step_{step}.pkl"),
                {"params": params, "opt_state": opt_state, "step": step, "cfg": dataclasses.asdict(cfg)
                 if dataclasses.is_dataclass(cfg) else None},
            )

    logger.close()
    return {"params": params, "metrics": last_metrics}
