"""Per-slot glimpse encoder + decoder.

Encoder: read a glimpse from the image at z_where, encode to a feature vector that is added
into z_what (dual-source head — see model/heads.py).

Decoder: z_what → small ConvT stack → patch with (appearance, mask_logit) channels.
The patch is later placed onto the canvas via stn_write.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from sim2real.model.stn import stn_read

Array = jnp.ndarray


class GlimpseEncoder(nn.Module):
    """Encode a (g, g, C) glimpse to a feature vector of dim `feat_dim`."""

    feat_dim: int = 128
    channels: tuple[int, ...] = (32, 64)

    @nn.compact
    def __call__(self, glimpse):
        x = glimpse
        for c in self.channels:
            x = nn.Conv(c, (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.gelu(x)
        x = x.reshape(-1)
        return nn.Dense(self.feat_dim)(x)


def read_glimpse(image: Array, z_where: Array, glimpse_size: int) -> Array:
    """Functional helper exposing `stn_read` for the slot pipeline."""
    return stn_read(image, z_where, glimpse_size)


class GlimpseDecoder(nn.Module):
    """Decode z_what → (g, g, 2) patch with appearance + mask_logit channels.

    We use a small MLP→reshape→ConvT stack. Architecture is deliberately tiny: the work is done
    by the slot transformer; the decoder just produces a patch.
    """

    glimpse_size: int = 16
    z_what_dim: int = 128
    channels: tuple[int, ...] = (64, 32)

    @nn.compact
    def __call__(self, z_what):
        # Start from a 4x4 feature map, upsample to glimpse_size via ConvT.
        base = 4
        x = nn.Dense(base * base * self.channels[0])(z_what)
        x = nn.gelu(x).reshape(base, base, self.channels[0])

        cur = base
        for c in self.channels[1:]:
            x = nn.ConvTranspose(c, (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.gelu(x)
            cur *= 2
        while cur < self.glimpse_size:
            x = nn.ConvTranspose(self.channels[-1], (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.gelu(x)
            cur *= 2
        # Crop / resize to exact glimpse_size.
        x = x[: self.glimpse_size, : self.glimpse_size]
        # Final 1x1 to 2 channels.
        x = nn.Conv(2, (1, 1))(x)
        appearance = nn.sigmoid(x[..., 0:1])              # (g, g, 1)
        mask_logit = x[..., 1:2]
        return appearance, mask_logit                     # both (g, g, 1)


class SegHead(nn.Module):
    """Dedicated per-slot segmentation head: independent of appearance.

    Takes [z_what; z_where_decoded] → small MLP → (g, g, 1) mask logit patch.
    """

    glimpse_size: int = 16
    hidden: int = 128

    @nn.compact
    def __call__(self, z_what, z_where):
        from jax import nn as jnn
        s = jnn.sigmoid(z_where[..., 0:1])
        txy = jnp.tanh(z_where[..., 1:3])
        feat = jnp.concatenate([z_what, s, txy], axis=-1)
        x = nn.Dense(self.hidden)(feat)
        x = nn.gelu(x)
        x = nn.Dense(self.glimpse_size * self.glimpse_size)(x)
        return x.reshape(self.glimpse_size, self.glimpse_size, 1)
