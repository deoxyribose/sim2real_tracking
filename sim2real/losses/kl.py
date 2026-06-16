"""Aggregate KL term across all priors.

Each call computes:
  L_KL = w_where  · KL_where(...)
       + w_pres   · KL_pres(...)
       + w_what_t · KL_what_temporal(...)
       + w_what_x · KL_what_cross(...)
       + w_style  · KL_style(...)

All inputs come from `ModelOut` (with `aux`); see `losses/losses.py` for the wiring.
"""

from __future__ import annotations

import jax.numpy as jnp

from sim2real.priors import pres as pres_p
from sim2real.priors import style as style_p
from sim2real.priors import what as what_p
from sim2real.priors import where as where_p
from sim2real.priors.registry import PriorConfig

Array = jnp.ndarray


def total_kl(
    z_where: Array,                # (T, N, 3)
    z_pres: Array,                 # (T, N)
    z_pres_logit: Array,           # (T, N)
    mu_w: Array,                   # (T, N, Zw)
    lv_w: Array,                   # (T, N, Zw)
    z_style_mu: Array,             # (Zs,)
    z_style_logvar: Array,         # (Zs,)
    alive_mask: Array,             # (T, N) — typically z_pres (soft) or GT alive
    discovery_score: Array | None, # (T, N) in [0, 1] or None
    priors_cfg: PriorConfig,
) -> tuple[Array, dict]:
    # We use the random-walk negative log-prob as a Dirac KL surrogate for z_where.
    kl_where = where_p.kl_walk(z_where[1:], z_where[:-1], alive_mask[1:], priors_cfg.where)

    kl_pres = pres_p.kl(z_pres, z_pres_logit, discovery_score, priors_cfg.pres)

    # Use posterior means for AR(1) chain (prev_mu_q is mu_q itself; treated as Dirac).
    kl_what_t = what_p.kl_temporal(mu_w, lv_w, mu_w, alive_mask, priors_cfg.what)
    kl_what_x = what_p.kl_cross_object(mu_w, lv_w, alive_mask, priors_cfg.what)

    kl_style = style_p.kl(z_style_mu, z_style_logvar)

    total = (
        priors_cfg.weight_where * kl_where
        + priors_cfg.weight_pres * kl_pres
        + priors_cfg.weight_what_temporal * kl_what_t
        + priors_cfg.weight_what_cross * kl_what_x
        + priors_cfg.weight_style * kl_style
    )
    return total, {
        "kl_where": kl_where,
        "kl_pres": kl_pres,
        "kl_what_t": kl_what_t,
        "kl_what_x": kl_what_x,
        "kl_style": kl_style,
    }
