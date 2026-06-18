"""Top-level loss aggregators: PretrainLoss (supervised) and AdaptLoss (unsupervised)."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from sim2real.losses.kl import total_kl
from sim2real.losses.matching import gather_along_slots, hungarian_per_frame
from sim2real.losses.recon import recon_mse
from sim2real.losses.supervised import (
    bce_from_logits,
    group_supervision_nll,
    group_temporal_kl,
    mask_loss,
    masked_mse,
)
from sim2real.priors.registry import PriorConfig
from sim2real.types import ModelOut, SimSample

Array = jnp.ndarray


@dataclass(frozen=True)
class PretrainLossConfig:
    lambda_recon: float = 1.0
    lambda_where: float = 1.0
    lambda_pres: float = 0.5
    lambda_mask: float = 1.0
    # KL is not used during supervised pretrain — the GT labels pin all latents directly.
    # Set >0 only if you want to lightly nudge posteriors toward the priors.
    lambda_kl: float = 0.0
    # Group latent losses (only meaningful when ModelConfig.n_groups > 1).
    # lambda_group: supervised NLL forcing slot_n → group_n (each instance its own group).
    # lambda_group_temp: symmetric KL between consecutive frames' g_post per slot (coherence).
    lambda_group: float = 0.0
    lambda_group_temp: float = 0.0


@dataclass(frozen=True)
class AdaptLossConfig:
    lambda_recon: float = 1.0
    lambda_kl: float = 0.05


def _match_video(out: ModelOut, sample: SimSample) -> Array:
    """Per-frame Hungarian matching on z_where. Returns perm of shape (T, N)."""
    T, N, _ = out.z_where.shape

    def per_frame(pred_zw, gt_zw, gt_pres):
        return hungarian_per_frame(pred_zw, gt_zw, gt_pres)

    return jax.vmap(per_frame)(out.z_where, sample.z_where, sample.z_pres)            # (T, N)


def _apply_perm(arr: Array, perm: Array) -> Array:
    """`arr` shape (T, N, ...), `perm` shape (T, N) → arr[perm] along axis 1 per frame."""
    return jax.vmap(gather_along_slots)(arr, perm)


def pretrain_loss(out: ModelOut, sample: SimSample, cfg: PretrainLossConfig,
                  prior_cfg: PriorConfig) -> tuple[Array, dict]:
    """Supervised pretrain loss. Single video (no batch dim)."""
    perm = _match_video(out, sample)
    z_where_matched = _apply_perm(out.z_where, perm)
    z_pres_matched = _apply_perm(out.z_pres, perm)
    z_pres_logit_matched = _apply_perm(out.aux["z_pres_logit"], perm)
    mu_w_matched = _apply_perm(out.aux["mu_w"], perm)
    lv_w_matched = _apply_perm(out.aux["lv_w"], perm)
    masks_pred_matched = _apply_perm(out.masks_pred, perm)

    L_recon = recon_mse(out.composite, sample.video)
    L_where = masked_mse(z_where_matched, sample.z_where, sample.z_pres)
    L_pres = bce_from_logits(z_pres_logit_matched, sample.z_pres)
    L_mask = mask_loss(masks_pred_matched, sample.masks, sample.z_pres)

    g_post_raw = out.aux.get("g_post")
    if g_post_raw is not None and g_post_raw.shape[-1] > 1:
        g_post_matched = _apply_perm(g_post_raw, perm)
        L_group = group_supervision_nll(g_post_matched, sample.z_pres)
        L_group_temp = group_temporal_kl(g_post_matched, sample.z_pres)
    else:
        L_group = jnp.array(0.0)
        L_group_temp = jnp.array(0.0)

    L_kl, kl_breakdown = total_kl(
        z_where_matched,
        z_pres_matched,
        z_pres_logit_matched,
        mu_w_matched,
        lv_w_matched,
        out.aux["z_style_mu"],
        out.aux["z_style_logvar"],
        sample.z_pres,
        discovery_score=None,
        priors_cfg=prior_cfg,
    )

    total = (
        cfg.lambda_recon * L_recon
        + cfg.lambda_where * L_where
        + cfg.lambda_pres * L_pres
        + cfg.lambda_mask * L_mask
        + cfg.lambda_kl * L_kl
        + cfg.lambda_group * L_group
        + cfg.lambda_group_temp * L_group_temp
    )
    metrics = {
        "loss": total,
        "L_recon": L_recon,
        "L_where": L_where,
        "L_pres": L_pres,
        "L_mask": L_mask,
        "L_group": L_group,
        "L_group_temp": L_group_temp,
        "L_kl": L_kl,
        **kl_breakdown,
    }
    return total, metrics


def adapt_loss(out: ModelOut, sample: SimSample, cfg: AdaptLossConfig,
               prior_cfg: PriorConfig) -> tuple[Array, dict]:
    """Unsupervised adaptation loss — no Hungarian, no GT-matched terms. Only recon + KL.

    The `sample` argument is still passed because we use sample.video as the recon target; GT
    latents are ignored.
    """
    L_recon = recon_mse(out.composite, sample.video)
    alive = out.z_pres  # use predicted presence as soft alive mask
    L_kl, kl_breakdown = total_kl(
        out.z_where,
        out.z_pres,
        out.aux["z_pres_logit"],
        out.aux["mu_w"],
        out.aux["lv_w"],
        out.aux["z_style_mu"],
        out.aux["z_style_logvar"],
        alive,
        discovery_score=None,
        priors_cfg=prior_cfg,
    )
    total = cfg.lambda_recon * L_recon + cfg.lambda_kl * L_kl
    return total, {"loss": total, "L_recon": L_recon, "L_kl": L_kl, **kl_breakdown}
