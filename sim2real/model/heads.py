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
    """z_pres head — Bernoulli logit, sampled via Gumbel-sigmoid with straight-through.

    When `image_attn=True`, does a cross-attention from `q` over the image feature grid,
    concatenates the resulting context with `q`, and feeds through the MLP. Motivation: the
    slot query alone is a compressed d=128 summary — this gives pres a direct look at the
    image so it can answer "is there really an object at my slot's location?" with pixel evidence.
    """

    hidden: int = 64
    depth: int = 1
    init_bias: float = -1.0   # prior bias toward "not present"
    image_attn: bool = False
    n_heads: int = 4

    @nn.compact
    def __call__(self, q, key, *, tau: float = 0.5, straight_through: bool = True, image_feats=None):
        x = q
        if self.image_attn and image_feats is not None:
            # image_feats: (h, w, d). Flatten spatial → sequence for attention.
            h, w, d = image_feats.shape
            kv = image_feats.reshape(h * w, d)
            # q: (d,) → (1, d) for MHA
            attn = nn.MultiHeadDotProductAttention(num_heads=self.n_heads, qkv_features=d)
            ctx = attn(q[None, :], kv).squeeze(0)                                        # (d,)
            x = jnp.concatenate([q, ctx], axis=-1)
        for _ in range(self.depth):
            x = nn.Dense(self.hidden)(x)
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
