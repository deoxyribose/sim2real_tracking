"""Image encoder: 3-stage stride-2 CNN → 2 ViT layers with sinusoidal 2D PE.

Operates on a single frame (H, W, C); the trainer vmaps over (B, T) externally.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from sim2real.model.nets import MultiHeadSelfAttention
from sim2real.model.posenc import sinusoidal_2d

Array = jnp.ndarray


class ConvStem(nn.Module):
    channels: tuple[int, ...] = (32, 64, 128)
    d_model: int = 256

    @nn.compact
    def __call__(self, x):
        # x: (H, W, C)
        for c in self.channels:
            x = nn.Conv(c, (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.GroupNorm(num_groups=8 if c >= 8 else c)(x)
            x = nn.gelu(x)
        x = nn.Conv(self.d_model, (1, 1))(x)
        return x  # (H/8, W/8, d_model)


class ViTBlock(nn.Module):
    d_model: int
    n_heads: int = 8
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, tokens):
        # tokens: (L, d_model)
        h = nn.LayerNorm()(tokens)
        h = MultiHeadSelfAttention(self.d_model, self.n_heads)(h)
        tokens = tokens + h
        h = nn.LayerNorm()(tokens)
        h = nn.Dense(self.d_model * self.mlp_ratio)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model)(h)
        return tokens + h


class FrameEncoder(nn.Module):
    """Single-frame encoder.

    Args:
      d_model: token dimensionality.
      n_vit_layers: number of ViT blocks after the stem.

    Returns:
      feat_grid: (H', W', d_model) where H' = H // 8.
      pool: (d_model,) mean pool of feat_grid.
    """

    d_model: int = 256
    n_vit_layers: int = 2
    stem_channels: tuple[int, ...] = (32, 64, 128)

    @nn.compact
    def __call__(self, image):
        feat = ConvStem(self.stem_channels, self.d_model)(image)                  # (h', w', d)
        h, w, d = feat.shape
        pe = sinusoidal_2d(h, w, d)
        tokens = (feat + pe).reshape(h * w, d)
        for _ in range(self.n_vit_layers):
            tokens = ViTBlock(self.d_model)(tokens)
        feat_grid = tokens.reshape(h, w, d)
        pool = jnp.mean(feat_grid, axis=(0, 1))
        return feat_grid, pool
