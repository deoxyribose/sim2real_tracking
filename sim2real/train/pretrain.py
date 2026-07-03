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
    # When True, model.stn_read/write uses GT z_where (and the residual head's anchor is GT[t-1])
    # while the predicted z_where is still produced and trained via L_where.
    teacher_force_zwhere: bool = False
    # When True, the alive/dormant gate that selects propagate vs discover uses GT z_pres[t-1].
    teacher_force_zpres: bool = False
    # SQAIR-style frame curriculum: start at `t_start` frames, ramp linearly to the simulator's
    # native T over `t_curriculum_steps`. Each T bump triggers a JIT recompile (~15s).
    t_curriculum: bool = False
    t_start: int = 3
    t_curriculum_steps: int = 25_000
    # Held-out eval hook. `eval_every=0` disables. Samples one fixed batch at init with
    # `eval_seed` (distinct from `seed`) and forwards through it every `eval_every` steps at
    # native T. Logs eval/recon, eval/psnr, eval/seg_iou, eval/silhouette_zwhat.
    eval_every: int = 0
    eval_batch_size: int = 4
    eval_seed: int = 424242
    # Matching curriculum. When > 0, switch loss_cfg.matching_mode to "once" at this step by
    # rebuilding train_step (one JIT recompile). Lets z_where descend under per-frame matching
    # first, then locks in temporal identity via match-once.
    match_once_after: int = 0


def train_step_factory(model, loss_cfg, prior_cfg, optimizer,
                       teacher_force_zwhere=False, teacher_force_zpres=False):
    """Build a jitted train_step closure.

    teacher_force_*: if True, pass the corresponding GT tensors from the batch into model.apply,
    so the residual anchor / discover gate / STN read are anchored on GT slot data while the
    predicted latents are still produced and supervised by the losses.
    """

    def model_forward_one(params, video, key, t_zw, t_zp, boot0):
        return model.apply(params, video, key,
                           teacher_zwhere=t_zw, teacher_zpres=t_zp,
                           bootstrap_zwhere0=boot0)

    def loss_fn(params, batch, key):
        keys = jax.random.split(key, batch.video.shape[0])
        t_zw = batch.z_where if teacher_force_zwhere else None
        t_zp = batch.z_pres if teacher_force_zpres else None
        # No SAVi bootstrap: model must learn its own z_where_init and residual head dynamics.

        def fwd_one(v, k, t_zw_i, t_zp_i):
            return model_forward_one(params, v, k, t_zw_i, t_zp_i, None)

        if t_zw is None and t_zp is None:
            outs = jax.vmap(lambda v, k: fwd_one(v, k, None, None))(batch.video, keys)
        elif t_zw is not None and t_zp is None:
            outs = jax.vmap(lambda v, k, z: fwd_one(v, k, z, None))(batch.video, keys, t_zw)
        elif t_zw is None and t_zp is not None:
            outs = jax.vmap(lambda v, k, p: fwd_one(v, k, None, p))(batch.video, keys, t_zp)
        else:
            outs = jax.vmap(lambda v, k, z, p: fwd_one(v, k, z, p))(batch.video, keys, t_zw, t_zp)

        def per_video(out, smp):
            total, metrics = pretrain_loss(out, smp, loss_cfg, prior_cfg)
            return total, metrics

        totals, metrics = jax.vmap(per_video)(outs, batch)
        loss = jnp.mean(totals)
        return loss, jax.tree.map(jnp.mean, metrics)

    @jax.jit
    def train_step(params, opt_state, batch, key):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, key)
        # Compute gradient norm BEFORE skipping — useful diagnostic even on bad batches.
        gnorm = optax.global_norm(grads)
        # NaN guard: if any grad leaf is non-finite, skip the update entirely.
        finite_per_leaf = jax.tree.map(lambda x: jnp.all(jnp.isfinite(x)), grads)
        all_finite = jax.tree.reduce(jnp.logical_and, finite_per_leaf)

        def good_update(_):
            updates, new_st = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_st

        def skip_update(_):
            return params, opt_state

        new_params, new_opt_state = jax.lax.cond(all_finite, good_update, skip_update, operand=None)
        metrics["grad_norm"] = gnorm
        metrics["skipped_nan"] = (~all_finite).astype(jnp.float32)
        return new_params, new_opt_state, loss, metrics

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
    train_step = train_step_factory(
        model, cfg.loss_cfg, cfg.prior_cfg, optimizer,
        teacher_force_zwhere=cfg.teacher_force_zwhere,
        teacher_force_zpres=cfg.teacher_force_zpres,
    )
    print(
        f"teacher_force_zwhere={cfg.teacher_force_zwhere}  "
        f"teacher_force_zpres={cfg.teacher_force_zpres}"
    )

    logger = Logger(cfg.run_dir)
    os.makedirs(os.path.join(cfg.run_dir, "ckpts"), exist_ok=True)

    # ---------------- Held-out eval hook ----------------
    eval_batch_m = None
    eval_forward = None
    if cfg.eval_every > 0:
        from sim2real.eval.recon import psnr as _psnr, ssim_simple as _ssim
        from sim2real.eval.seg_iou import matched_seg_iou as _matched_iou
        from sim2real.eval.disentangle import silhouette_zwhat as _silhouette
        import numpy as _np

        eval_batcher = SimBatcher(cfg.sim_kind, cfg.eval_batch_size, cfg.sim_cfg)
        eval_key = jax.random.key(cfg.eval_seed)
        eval_batch = eval_batcher.jit_sample()(eval_key)
        eval_batch_m = slice_to_model(eval_batch)

        @jax.jit
        def _eval_forward(params_now, video, key):
            keys = jax.random.split(key, video.shape[0])
            def one(v, k):
                return model.apply(params_now, v, k)
            return jax.vmap(one)(video, keys)
        eval_forward = _eval_forward

        def run_eval(params_now, step):
            k = jax.random.key(cfg.eval_seed + step)
            outs = eval_forward(params_now, eval_batch_m.video, k)
            # Recon
            mse = float(jnp.mean((outs.composite - eval_batch_m.video) ** 2))
            psnr_val = float(_np.mean([_psnr(outs.composite[i], eval_batch_m.video[i])
                                       for i in range(cfg.eval_batch_size)]))
            ssim_val = float(_np.mean([_ssim(outs.composite[i], eval_batch_m.video[i])
                                       for i in range(cfg.eval_batch_size)]))
            # Seg IoU (Hungarian per-sample, host-side)
            iou_val = float(_np.mean([
                float(_matched_iou(
                    outs.z_where[i], outs.masks_pred[i],
                    eval_batch_m.z_where[i], eval_batch_m.masks[i], eval_batch_m.z_pres[i],
                )) for i in range(cfg.eval_batch_size)
            ]))
            # Silhouette on z_what (slot index = identity)
            pred_zwhat = _np.asarray(outs.z_what).reshape(-1, *outs.z_what.shape[2:])
            gt_pres = _np.asarray(eval_batch_m.z_pres).reshape(-1, *eval_batch_m.z_pres.shape[2:])
            Nm_local = eval_batch_m.z_pres.shape[-1]
            gt_ids = _np.broadcast_to(_np.arange(Nm_local)[None, None, :], eval_batch_m.z_pres.shape)
            gt_ids = gt_ids.reshape(-1, Nm_local)
            sil = _silhouette(pred_zwhat, gt_pres, gt_ids)
            logger.scalar("eval/recon_mse", mse, step)
            logger.scalar("eval/psnr", psnr_val, step)
            logger.scalar("eval/ssim", ssim_val, step)
            logger.scalar("eval/seg_iou", iou_val, step)
            logger.scalar("eval/silhouette_zwhat", float(sil) if sil == sil else 0.0, step)
            print(
                f"[eval] step {step:6d}  recon_mse {mse:.4f}  psnr {psnr_val:.2f}  "
                f"ssim {ssim_val:.3f}  iou {iou_val:.3f}  silhouette {sil:.3f}",
                flush=True,
            )
    # ---------------------------------------------------

    def current_T(step: int) -> int:
        """SQAIR-style frame curriculum. Returns the current sequence length."""
        if not cfg.t_curriculum:
            return int(sample.video.shape[1])
        T_max = int(sample.video.shape[1])
        if step >= cfg.t_curriculum_steps:
            return T_max
        frac = step / max(cfg.t_curriculum_steps, 1)
        T = int(cfg.t_start + frac * (T_max - cfg.t_start))
        return max(cfg.t_start, min(T_max, T))

    def slice_T(batch_m, T: int):
        from sim2real.types import SimSample
        return SimSample(
            video=batch_m.video[:, :T],
            z_where=batch_m.z_where[:, :T],
            z_pres=batch_m.z_pres[:, :T],
            z_style=batch_m.z_style,
            masks=batch_m.masks[:, :T],
            z_what=batch_m.z_what,
            meta=batch_m.meta,
        )

    rng_iter = rng
    t0 = time.time()
    last_metrics = None
    last_T = -1
    skipped_nan = 0
    matching_switched = False
    for step in range(1, cfg.n_steps + 1):
        rng_iter, k_batch, k_step = jax.random.split(rng_iter, 3)
        batch = jit_sample(k_batch)
        batch_m = slice_to_model(batch)
        T_now = current_T(step)
        batch_m = slice_T(batch_m, T_now)

        # Matching curriculum: at step cfg.match_once_after, switch to match-once by
        # rebuilding train_step with a new loss_cfg. Triggers one JIT recompile.
        if (cfg.match_once_after > 0 and step == cfg.match_once_after
                and not matching_switched):
            new_loss_cfg = dataclasses.replace(cfg.loss_cfg, matching_mode="once")
            train_step = train_step_factory(
                model, new_loss_cfg, cfg.prior_cfg, optimizer,
                teacher_force_zwhere=cfg.teacher_force_zwhere,
                teacher_force_zpres=cfg.teacher_force_zpres,
            )
            matching_switched = True
            print(f"[matching-curriculum] step {step}: switched to match-once "
                  f"(rebuilding train_step)", flush=True)
        if T_now != last_T:
            print(f"[curriculum] step {step}: T = {T_now}", flush=True)
            last_T = T_now

        params, opt_state, loss, metrics = train_step(params, opt_state, batch_m, k_step)
        if bool(metrics["skipped_nan"] > 0.5):
            skipped_nan += 1
            print(f"[nan-guard] step {step}: non-finite grad detected — update skipped", flush=True)
        last_metrics = metrics

        if step % cfg.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            print(
                f"step {step:6d}  T={T_now:2d}  loss {float(loss):.4f}  recon {float(metrics['L_recon']):.4f}  "
                f"where {float(metrics['L_where']):.4f}  pres {float(metrics['L_pres']):.4f}  "
                f"mask {float(metrics['L_mask']):.4f}  "
                f"appearG {float(metrics.get('L_appear_glimpse', 0.0)):.4f}  "
                f"contr {float(metrics.get('L_slot_contrast', 0.0)):.3f}  "
                f"gnorm {float(metrics['grad_norm']):.2f}  nan_skips {skipped_nan}  "
                f"({elapsed:.1f}s)",
                flush=True,
            )
            for k, v in metrics.items():
                logger.scalar(f"train/{k}", v, step)
            logger.scalar("train/lr", float(lr_schedule(step)), step)
            logger.scalar("train/T", T_now, step)
            logger.scalar("train/skipped_nan", skipped_nan, step)

        if cfg.eval_every > 0 and (step % cfg.eval_every == 0 or step == 1):
            run_eval(params, step)

        if step % cfg.ckpt_every == 0 or step == cfg.n_steps:
            ckpt_save(
                os.path.join(cfg.run_dir, "ckpts", f"step_{step}.pkl"),
                {"params": params, "opt_state": opt_state, "step": step, "cfg": dataclasses.asdict(cfg)
                 if dataclasses.is_dataclass(cfg) else None},
            )

    logger.close()
    return {"params": params, "metrics": last_metrics}
