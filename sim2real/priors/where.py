"""z_where prior: Gaussian random walk on (s_raw, tx_raw, ty_raw).

At t=0, prior is N(0, σ_init²·I). For t≥1, p(z_t | z_{t-1}) = N(z_{t-1}, Σ_step).
Σ_step is diagonal with per-component σ from the simulator config.

Implementation notes:
- log_prob is summed over (T, N) (alive slots only — multiply by `alive_mask` before summing).
- The KL between a Gaussian posterior with (mu_q, logvar_q) and this random-walk prior is computed
  as if the prior mean is `prev_z_where` and the prior log-variance is `2 log σ_step`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from sim2real.priors.base import gaussian_kl

Array = jnp.ndarray


@dataclass(frozen=True)
class WherePriorConfig:
    # 5-dim z_where = (sx_raw, sy_raw, theta_raw, tx_raw, ty_raw).
    sigma_init: tuple = (0.5, 0.5, 0.3, 1.0, 1.0)
    sigma_step: tuple = (0.05, 0.05, 0.03, 0.1, 0.1)


def log_prob(z_where: Array, alive_mask: Array, cfg: WherePriorConfig) -> Array:
    """
    Args:
      z_where: (T, N, 3)
      alive_mask: (T, N) in {0, 1}.

    Returns: scalar sum of log p over alive slots.
    """
    T = z_where.shape[0]
    sig_init = jnp.asarray(cfg.sigma_init)
    sig_step = jnp.asarray(cfg.sigma_step)

    # t=0
    lp0 = -0.5 * jnp.sum(
        (z_where[0] / sig_init) ** 2 + jnp.log(2.0 * jnp.pi * sig_init ** 2),
        axis=-1,
    )                                                                                # (N,)
    lp0 = jnp.sum(lp0 * alive_mask[0])

    # t>=1
    delta = z_where[1:] - z_where[:-1]                                               # (T-1, N, 3)
    lp_t = -0.5 * jnp.sum(
        (delta / sig_step) ** 2 + jnp.log(2.0 * jnp.pi * sig_step ** 2),
        axis=-1,
    )                                                                                # (T-1, N)
    lp_t = jnp.sum(lp_t * alive_mask[1:])
    return lp0 + lp_t


def kl_walk(z_where_t: Array, z_where_tm1: Array, alive_mask_t: Array,
            cfg: WherePriorConfig) -> Array:
    """KL when treating z_where as a Dirac posterior — degenerates to the log-prob negation.

    For variational z_where (not used by default), see `kl_gaussian` below.
    """
    sig_step = jnp.asarray(cfg.sigma_step)
    delta = z_where_t - z_where_tm1
    per_slot = 0.5 * jnp.sum((delta / sig_step) ** 2, axis=-1)
    return jnp.sum(per_slot * alive_mask_t)


def kl_gaussian(mu_q: Array, logvar_q: Array, mu_p: Array, alive_mask: Array,
                cfg: WherePriorConfig) -> Array:
    """Optional variational KL: KL(q(z_w_t) || N(mu_p, σ_step)) summed over alive slots."""
    sig_step = jnp.asarray(cfg.sigma_step)
    logvar_p = jnp.broadcast_to(2.0 * jnp.log(sig_step), mu_p.shape)
    per_slot = gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)
    return jnp.sum(per_slot * alive_mask)
