"""z_style prior: i.i.d. unit Gaussian per video."""

from __future__ import annotations

import jax.numpy as jnp

from sim2real.priors.base import gaussian_kl

Array = jnp.ndarray


def kl(mu_q: Array, logvar_q: Array) -> Array:
    """Standard KL against N(0, I)."""
    mu_p = jnp.zeros_like(mu_q)
    logvar_p = jnp.zeros_like(logvar_q)
    return gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)


def log_prob(z_style: Array) -> Array:
    Zs = z_style.shape[-1]
    return -0.5 * (jnp.sum(z_style ** 2) + Zs * jnp.log(2.0 * jnp.pi))
