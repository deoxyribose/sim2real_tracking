"""Encoder: canonical clip + energy map → spatial feature map.

Input: (B, T, H, W) + (B, H, W).
Stack: temporal frames go in as channels. Energy prepended as one extra channel.
      → (B, H, W, T+1)
Then a ConvStem downsamples to (B, H', W', d_model) at stride 8.
Position embedding is added and features are flattened to (B, L, d_model).

Kept intentionally shallow — the interesting representation happens in the slot
attention. This is just enough to give slot attention a decent 32×32 feature grid.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class ConvStem(nn.Module):
    d_model: int = 128
    channels: tuple[int, ...] = (32, 64, 128)
    strides: tuple[int, ...] = (2, 2, 2)  # total 8×

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (B, H, W, C_in). Returns (B, H//8, W//8, d_model)."""
        for i, (c, s) in enumerate(zip(self.channels, self.strides)):
            x = nn.Conv(features=c, kernel_size=(3, 3), strides=(s, s), padding="SAME", name=f"conv{i}")(x)
            x = nn.GroupNorm(num_groups=8, name=f"gn{i}")(x)
            x = jax.nn.gelu(x)
        x = nn.Conv(features=self.d_model, kernel_size=(1, 1), name="proj")(x)
        return x


class Encoder(nn.Module):
    d_model: int = 128
    conv_channels: tuple[int, ...] = (32, 64, 128)
    conv_strides: tuple[int, ...] = (2, 2, 2)

    @nn.compact
    def __call__(self, clip: jnp.ndarray, energy: jnp.ndarray) -> jnp.ndarray:
        """clip: (B, T, H, W). energy: (B, H, W). Returns features (B, L, d_model) + shape (H', W')."""
        B, T, H, W = clip.shape
        # Frames → channels; energy prepended
        x = jnp.concatenate([energy[:, None], clip], axis=1)   # (B, T+1, H, W)
        x = jnp.moveaxis(x, 1, -1)                              # (B, H, W, T+1)
        feats = ConvStem(d_model=self.d_model, channels=self.conv_channels, strides=self.conv_strides,
                         name="stem")(x)
        Bp, Hp, Wp, D = feats.shape
        # Learned 2D positional embedding
        pos_h = self.param("pos_h", nn.initializers.normal(0.02), (1, Hp, 1, D))
        pos_w = self.param("pos_w", nn.initializers.normal(0.02), (1, 1, Wp, D))
        feats = feats + pos_h + pos_w
        feats_flat = feats.reshape(B, Hp * Wp, D)
        return feats_flat, (Hp, Wp)
