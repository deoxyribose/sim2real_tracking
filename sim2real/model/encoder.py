"""Image encoders: ConvStem + optional ViT (default), or a U-Net backbone.

Operates on a single frame (H, W, C); the trainer vmaps over (B, T) externally.
Both encoders return `(feat_grid, pool)` where `feat_grid` has shape (H', W', d_model).
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from sim2real.model.nets import MultiHeadSelfAttention
from sim2real.model.posenc import sinusoidal_2d

Array = jnp.ndarray


class ConvStem(nn.Module):
    channels: tuple[int, ...] = (32, 64, 128)
    d_model: int = 256
    # Per-stage strides. Length must match `channels`. Default is all stride-2 (total stride 8).
    # Use e.g. (2, 2, 1) for total stride 4 → 32x32 feature grid on 128 input (halves pixel
    # quantum from 8 to 4, matching SLATE/STEVE/SlotFormer).
    strides: tuple[int, ...] = (2, 2, 2)

    @nn.compact
    def __call__(self, x):
        # x: (H, W, C)
        assert len(self.strides) == len(self.channels), "strides must match channels"
        for c, s in zip(self.channels, self.strides):
            x = nn.Conv(c, (3, 3), strides=(s, s), padding="SAME")(x)
            x = nn.GroupNorm(num_groups=8 if c >= 8 else c)(x)
            x = nn.gelu(x)
        x = nn.Conv(self.d_model, (1, 1))(x)
        return x  # (H/prod(strides), W/prod(strides), d_model)


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
    stem_strides: tuple[int, ...] = (2, 2, 2)

    @nn.compact
    def __call__(self, image):
        feat = ConvStem(self.stem_channels, self.d_model, self.stem_strides)(image)  # (h', w', d)
        h, w, d = feat.shape
        pe = sinusoidal_2d(h, w, d)
        tokens = (feat + pe).reshape(h * w, d)
        for _ in range(self.n_vit_layers):
            tokens = ViTBlock(self.d_model)(tokens)
        feat_grid = tokens.reshape(h, w, d)
        pool = jnp.mean(feat_grid, axis=(0, 1))
        return feat_grid, pool


class UNetEncoder(nn.Module):
    """U-Net backbone: 128x128 → down to 8x8 bottleneck → up to `out_res` with skip connections.

    Multi-scale features by construction: bottleneck sees global context, up-path recovers
    spatial detail via skip concatenations from the corresponding down-stage feature map.
    Memory is O(H*W*d) per layer — no O(L^2) attention matrix, unlike the ViT encoder.

    Args:
      d_model: final per-token dimensionality (1x1 conv projection on the up-path output).
      down_channels: channel widths for each down stage. Length K means K down stages, so
        input at 128 goes down by 2^(K-1) (first stage does not downsample). Default (24, 48,
        96, 128, 192) → resolutions 128, 64, 32, 16, 8.
      n_up_stages: how many upsample steps after the bottleneck. Determines output grid
        resolution: `out_res = 128 / 2^(K-1-n_up_stages)`. n_up_stages=2 → 32x32 (typical);
        n_up_stages=3 → 64x64 (for small-object recovery).
      blocks_per_stage: number of Conv→GN→GELU blocks in each stage (both down and up).
    """

    d_model: int = 192
    down_channels: tuple[int, ...] = (24, 48, 96, 128, 192)
    n_up_stages: int = 2
    blocks_per_stage: int = 2

    @nn.compact
    def __call__(self, image):
        assert self.n_up_stages < len(self.down_channels), \
            f"n_up_stages={self.n_up_stages} must be < len(down_channels)={len(self.down_channels)}"

        x = image
        skips = []
        for i, c in enumerate(self.down_channels):
            for _ in range(self.blocks_per_stage):
                x = nn.Conv(c, (3, 3), padding="SAME")(x)
                x = nn.GroupNorm(num_groups=min(8, c))(x)
                x = nn.gelu(x)
            skips.append(x)
            if i < len(self.down_channels) - 1:
                x = nn.Conv(c, (3, 3), strides=(2, 2), padding="SAME")(x)  # halve resolution

        pool = jnp.mean(x, axis=(0, 1))  # bottleneck global pool

        for j in range(self.n_up_stages):
            H, W, C = x.shape
            x = jax.image.resize(x, (H * 2, W * 2, C), method="nearest")
            skip = skips[-2 - j]
            x = jnp.concatenate([x, skip], axis=-1)
            c_out = self.down_channels[-2 - j]
            for _ in range(self.blocks_per_stage):
                x = nn.Conv(c_out, (3, 3), padding="SAME")(x)
                x = nn.GroupNorm(num_groups=min(8, c_out))(x)
                x = nn.gelu(x)

        feat_grid = nn.Conv(self.d_model, (1, 1))(x)
        return feat_grid, pool
