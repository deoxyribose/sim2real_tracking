"""Feature-wise Linear Modulation (FiLM / AdaLN) helpers.

Given a global latent `z_style`, produce per-channel (γ, β) modulation parameters which scale
and shift normalized features: `out = (1 + γ) * x + β`.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

Array = jnp.ndarray


class FiLMParams(nn.Module):
    """Produce (gamma, beta) of dim `out_dim` from a latent vector."""

    out_dim: int

    @nn.compact
    def __call__(self, z):
        h = nn.Dense(self.out_dim * 2, kernel_init=nn.initializers.zeros)(z)
        gamma, beta = jnp.split(h, 2, axis=-1)
        return gamma, beta


def apply_film(x: Array, gamma: Array, beta: Array) -> Array:
    """x has last-dim = out_dim. gamma/beta broadcast over leading dims."""
    # Insert broadcasting axes
    while gamma.ndim < x.ndim:
        gamma = gamma[None]
        beta = beta[None]
    return (1.0 + gamma) * x + beta
