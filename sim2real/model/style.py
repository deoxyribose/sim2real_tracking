"""z_style head — global per-video latent.

Pools encoder features over (T, H', W') → MLP → (μ, log σ) of dim Zs.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jnp.ndarray


class StyleEncoder(nn.Module):
    z_style_dim: int = 32
    hidden: int = 128

    @nn.compact
    def __call__(self, pooled_feats, key):
        """pooled_feats: (T, d_model) — per-frame mean-pooled features.

        Returns (sample, mu, logvar) of shape (Zs,).
        """
        x = jnp.mean(pooled_feats, axis=0)                                     # (d_model,)
        x = nn.Dense(self.hidden)(x)
        x = nn.gelu(x)
        mu = nn.Dense(self.z_style_dim)(x)
        logvar = nn.Dense(
            self.z_style_dim,
            bias_init=nn.initializers.constant(-2.0),
        )(x)
        logvar = jnp.clip(logvar, -8.0, 2.0)
        eps = jax.random.normal(key, mu.shape)
        sample = mu + jnp.exp(0.5 * logvar) * eps
        return sample, mu, logvar
