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

from sim2real.losses.feat_recon import FeatDecoder, feat_recon_mse
from sim2real.losses.losses import AdaptLossConfig, adapt_loss
from sim2real.model.encoder import FrameEncoder
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
    savi_bootstrap: bool = False
    # DINOSAUR-lite: replace pixel MSE with per-pixel-feature MSE against a frozen target
    # encoder snapshot. Prevents STN opt-out collapse (target features have per-pixel structure).
    use_feat_recon: bool = False
    lambda_feat_recon: float = 1.0


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

    # Feat-recon: freeze a snapshot of the encoder params for target-feature computation,
    # and instantiate a new FeatDecoder module for the broadcast decode.
    if cfg.use_feat_recon:
        target_encoder = FrameEncoder(
            d_model=cfg.model_cfg.d_model,
            n_vit_layers=cfg.model_cfg.n_vit_layers,
            stem_channels=tuple(cfg.model_cfg.stem_channels),
            stem_strides=tuple(cfg.model_cfg.stem_strides),
        )
        # Encoder subtree of the loaded params — frozen snapshot, never trained.
        target_encoder_params = {"params": {"encoder": params["params"]["encoder"]}}
        # Wrap encoder call to match apply signature: apply(params, video[t]) -> (feat_grid, pooled)
        def target_encode(v):
            # v: (T, H, W, C) → (T, h, w, d)
            feats, _pools = jax.vmap(lambda vt: target_encoder.apply(
                {"params": target_encoder_params["params"]["encoder"]}, vt
            ))(v)
            return feats

        # Add feat_decoder params into `params` under a new top-level 'feat_decoder' key.
        # If loaded ckpt already contains it (warm restart), keep those params.
        feat_decoder = FeatDecoder(
            z_what_dim=cfg.model_cfg.z_what_dim, feat_dim=cfg.model_cfg.d_model,
        )
        if "feat_decoder" in params:
            print(f"reusing feat_decoder params from ckpt")
        else:
            rng_fd = jax.random.fold_in(init_key, 42)
            dummy_zwhat = jnp.zeros((cfg.model_cfg.n_max, cfg.model_cfg.z_what_dim))
            dummy_zwhere = jnp.zeros((cfg.model_cfg.n_max, 5))
            dummy_zpres = jnp.zeros((cfg.model_cfg.n_max,))
            fd_params = feat_decoder.init(rng_fd, dummy_zwhat, dummy_zwhere, dummy_zpres, 8, 8)
            params = {**params, "feat_decoder": fd_params}
    else:
        target_encode = None
        feat_decoder = None

    base_opt, lr_schedule = adamw_cosine(cfg.lr_peak, cfg.n_steps, cfg.warmup_steps)
    optimizer, mask = make_optimizer_with_freeze(base_opt, params, cfg.freeze_patterns)
    opt_state = optimizer.init(params)

    logger = Logger(cfg.run_dir)
    os.makedirs(os.path.join(cfg.run_dir, "ckpts"), exist_ok=True)

    def per_video(out, smp):
        total, m = adapt_loss(out, smp, cfg.loss_cfg, cfg.prior_cfg)
        return total, m

    savi_bootstrap = cfg.savi_bootstrap

    use_feat = cfg.use_feat_recon
    lambda_feat = cfg.lambda_feat_recon

    @jax.jit
    def train_step(p, st, b, k):
        def loss_fn(p):
            keys = jax.random.split(k, b.video.shape[0])
            boot0 = b.z_where[:, 0] if savi_bootstrap else None

            def one(v, kk, boot):
                # Extract model params (without feat_decoder subtree) for the SlotVideoModel.
                if use_feat:
                    p_model = {k_: v_ for k_, v_ in p.items() if k_ != "feat_decoder"}
                else:
                    p_model = p
                return model.apply(p_model, v, kk, bootstrap_zwhere0=boot)

            in_axes = (0, 0, 0 if boot0 is not None else None)
            outs = jax.vmap(one, in_axes=in_axes)(b.video, keys, boot0)

            if use_feat:
                # Feat-recon added on top of standard pixel-MSE recon.
                # target_feats: (B, T, h, w, d) via frozen target encoder
                target_feats_all = jax.vmap(target_encode)(b.video)                        # (B, T, h, w, d)
                H_feat, W_feat = target_feats_all.shape[2], target_feats_all.shape[3]

                def decode_one_video(z_what, z_where, z_pres):
                    def one_frame(zw, zh, zp):
                        return feat_decoder.apply(p["feat_decoder"], zw, zh, zp, H_feat, W_feat)
                    return jax.vmap(one_frame)(z_what, z_where, z_pres)

                pred_feats_all = jax.vmap(decode_one_video)(
                    outs.z_what, outs.z_where, outs.z_pres
                )                                                                          # (B, T, h, w, d)

                L_feat = feat_recon_mse(pred_feats_all, target_feats_all)
                mean_pres = jnp.mean(outs.z_pres)
                L_pres_alive = jnp.square(
                    jnp.maximum(cfg.loss_cfg.target_alive_rate - mean_pres, 0.0)
                )

                # Base pixel-MSE recon (via existing adapt_loss). lambda_recon may be 0 to use
                # feat-recon only.
                totals_base, metrics_base = jax.vmap(per_video)(outs, b)
                L_recon_pix = jnp.mean(totals_base)

                total = L_recon_pix + lambda_feat * L_feat + cfg.loss_cfg.lambda_alive * L_pres_alive
                metrics = {
                    "loss": total, "L_recon": jnp.mean(metrics_base["L_recon"]),
                    "L_feat": L_feat, "L_kl": jnp.mean(metrics_base["L_kl"]),
                    "L_alive": L_pres_alive, "mean_pres": mean_pres,
                }
                return total, metrics
            else:
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
            alive_metric = metrics.get("L_alive", 0.0)
            feat_metric = metrics.get("L_feat", 0.0)
            print(
                f"adapt step {step:6d}  loss {float(loss):.4f}  recon {float(metrics['L_recon']):.4f}  "
                f"feat {float(feat_metric):.4f}  KL {float(metrics['L_kl']):.4f}  alive {float(alive_metric):.4f}  "
                f"gnorm {float(metrics['grad_norm']):.2f}  ({elapsed:.1f}s)"
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
