"""Top-level loss aggregators: PretrainLoss (supervised) and AdaptLoss (unsupervised)."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from sim2real.losses.kl import total_kl
from sim2real.losses.matching import gather_along_slots, hungarian_per_frame
from sim2real.losses.recon import recon_mse
from sim2real.losses.supervised import bce_one_sided, mask_loss, masked_mse
from sim2real.priors.registry import PriorConfig
from sim2real.types import ModelOut, SimSample

Array = jnp.ndarray


@dataclass(frozen=True)
class PretrainLossConfig:
    lambda_recon: float = 1.0
    lambda_where: float = 1.0
    lambda_pres: float = 0.5
    lambda_mask: float = 1.0
    lambda_kl: float = 0.01


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
    L_pres = bce_one_sided(z_pres_matched, sample.z_pres)
    L_mask = mask_loss(masks_pred_matched, sample.masks, sample.z_pres)

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
    )
    metrics = {
        "loss": total,
        "L_recon": L_recon,
        "L_where": L_where,
        "L_pres": L_pres,
        "L_mask": L_mask,
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
