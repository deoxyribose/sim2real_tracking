"""Latent heads: z_where (residual, zero-init), z_pres (Gumbel-sigmoid + ST), z_what (dual-source).

All heads are single-slot — caller vmaps over slots.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jnp.ndarray


def _gumbel_sigmoid(key, logits, tau, straight_through=True):
    """Sample a relaxed-Bernoulli via Gumbel-sigmoid. With ST, forward = hard 0/1, backward = soft."""
    g1 = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape, minval=1e-6, maxval=1.0 - 1e-6) + 1e-12) + 1e-12)
    k2 = jax.random.fold_in(key, 1)
    g2 = -jnp.log(-jnp.log(jax.random.uniform(k2, logits.shape, minval=1e-6, maxval=1.0 - 1e-6) + 1e-12) + 1e-12)
    soft = jax.nn.sigmoid((logits + g1 - g2) / tau)
    if not straight_through:
        return soft
    hard = (soft > 0.5).astype(soft.dtype)
    return hard + jax.lax.stop_gradient(hard - soft) * 0.0 + (soft - jax.lax.stop_gradient(soft))


class WhereHead(nn.Module):
    """z_where residual head with zero-init final layer (v17 lesson).

    Now 5-dim: (sx_raw, sy_raw, theta_raw, tx_raw, ty_raw). Returns
    prev_z_where + scale * tanh(W·Q), W initialized to zero so the model starts at identity.
    """

    scale: float = 0.5
    hidden: int = 64
    dim: int = 5

    @nn.compact
    def __call__(self, q, prev_z_where):
        x = nn.Dense(self.hidden)(q)
        x = nn.gelu(x)
        delta = nn.Dense(self.dim, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)(x)
        return prev_z_where + self.scale * jnp.tanh(delta)


class PresHead(nn.Module):
    """z_pres head — Bernoulli logit, sampled via Gumbel-sigmoid with straight-through."""

    hidden: int = 64
    init_bias: float = -1.0   # prior bias toward "not present"

    @nn.compact
    def __call__(self, q, key, *, tau: float = 0.5, straight_through: bool = True):
        x = nn.Dense(self.hidden)(q)
        x = nn.gelu(x)
        logit = nn.Dense(
            1,
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=nn.initializers.constant(self.init_bias),
        )(x).squeeze(-1)
        sample = _gumbel_sigmoid(key, logit, tau, straight_through)
        return sample, logit


class WhatHead(nn.Module):
    """Dual-source z_what head: main MLP from slot query + glimpse-skip MLP, **added** (v18 lesson).

    The mean (μ) and log-variance are returned so a variational posterior is available; the
    sampled z_what is drawn during training. The trainer may opt to use the mean directly.
    """

    z_what_dim: int = 128
    hidden: int = 128

    @nn.compact
    def __call__(self, q, glimpse_feat, key):
        # Main branch from slot query.
        x = nn.Dense(self.hidden)(q)
        x = nn.gelu(x)
        mu_main = nn.Dense(self.z_what_dim)(x)
        logvar_main = nn.Dense(self.z_what_dim, bias_init=nn.initializers.constant(-2.0))(x)

        # Skip branch from glimpse features.
        s = nn.Dense(self.hidden)(glimpse_feat)
        s = nn.gelu(s)
        mu_skip = nn.Dense(self.z_what_dim, kernel_init=nn.initializers.zeros)(s)

        mu = mu_main + mu_skip
        logvar = jnp.clip(logvar_main, -8.0, 2.0)
        eps = jax.random.normal(key, mu.shape)
        sample = mu + jnp.exp(0.5 * logvar) * eps
        return sample, mu, logvar
