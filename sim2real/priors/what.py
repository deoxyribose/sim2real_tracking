"""z_what prior: AR(1) temporal smoothness + cross-object Gaussian KL toward per-video mean.

Temporal:  z_what_t ~ N(alpha · z_what_{t-1}, σ_step²·I) for t≥1; z_what_0 ~ N(0, σ_init²·I).
Cross-obj: q(z_what_n) ~ N(μ_q, σ_q); prior over each slot's z_what across the video is
           N(μ_video, σ_cross²·I) where μ_video = stop_grad(mean over alive slots' μ_q).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from sim2real.priors.base import gaussian_kl

Array = jnp.ndarray


@dataclass(frozen=True)
class WhatPriorConfig:
    sigma_init: float = 1.0
    sigma_step: float = 0.2
    alpha: float = 1.0          # AR(1) coefficient (1.0 = random walk)
    sigma_cross: float = 0.5    # cross-object spread


def log_prob_temporal(z_what: Array, alive_mask: Array, cfg: WhatPriorConfig) -> Array:
    """log p(z_what_{0:T} | priors). z_what: (T, N, Zw); alive_mask: (T, N)."""
    T, N, Zw = z_what.shape
    sig0_sq = cfg.sigma_init ** 2
    sigs_sq = cfg.sigma_step ** 2

    lp0 = -0.5 * (jnp.sum(z_what[0] ** 2, axis=-1) / sig0_sq + Zw * jnp.log(2.0 * jnp.pi * sig0_sq))
    lp0 = jnp.sum(lp0 * alive_mask[0])

    delta = z_what[1:] - cfg.alpha * z_what[:-1]                                     # (T-1, N, Zw)
    lp_t = -0.5 * (jnp.sum(delta ** 2, axis=-1) / sigs_sq + Zw * jnp.log(2.0 * jnp.pi * sigs_sq))
    lp_t = jnp.sum(lp_t * alive_mask[1:])
    return lp0 + lp_t


def kl_temporal(mu_q: Array, logvar_q: Array, prev_mu_q: Array,
                alive_mask: Array, cfg: WhatPriorConfig) -> Array:
    """KL of the Gaussian posterior at each step against the AR(1) prior.

    `prev_mu_q` is the posterior mean at t-1 (we treat it as Dirac for the AR(1) chain — i.e.,
    the prior at step t is N(alpha · mu_{t-1}, σ_step²·I)). At t=0, the prior is N(0, σ_init²·I).

    All inputs (T, N, Zw) or (T, N).
    """
    T, N, Zw = mu_q.shape
    # t = 0
    mu_p0 = jnp.zeros_like(mu_q[0])
    logvar_p0 = jnp.broadcast_to(jnp.log(cfg.sigma_init ** 2), mu_q[0].shape)
    kl0 = gaussian_kl(mu_q[0], logvar_q[0], mu_p0, logvar_p0)                        # (N,)
    kl0 = jnp.sum(kl0 * alive_mask[0])

    # t >= 1
    mu_p = cfg.alpha * prev_mu_q[1:]                                                 # (T-1, N, Zw)
    logvar_p = jnp.broadcast_to(jnp.log(cfg.sigma_step ** 2), mu_q[1:].shape)
    kl_rest = gaussian_kl(mu_q[1:], logvar_q[1:], mu_p, logvar_p)
    kl_rest = jnp.sum(kl_rest * alive_mask[1:])
    return kl0 + kl_rest


def kl_cross_object(mu_q: Array, logvar_q: Array, alive_mask: Array,
                    cfg: WhatPriorConfig) -> Array:
    """Cross-object KL: each slot's z_what is regularized toward the per-video mean.

    Args:
      mu_q, logvar_q: (T, N, Zw)
      alive_mask: (T, N)

    Returns: scalar sum over (T, N) alive slots.
    """
    # Compute per-video mean across alive slots and frames (stop_grad to avoid feedback loop).
    weights = alive_mask[..., None]                                                  # (T, N, 1)
    total = jnp.sum(weights)
    mu_video = jax.lax.stop_gradient(
        jnp.sum(mu_q * weights, axis=(0, 1)) / (total + 1e-6)
    )                                                                                # (Zw,)
    logvar_p = jnp.broadcast_to(jnp.log(cfg.sigma_cross ** 2), mu_q.shape)
    mu_p = jnp.broadcast_to(mu_video, mu_q.shape)
    per = gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)                                # (T, N)
    return jnp.sum(per * alive_mask)
