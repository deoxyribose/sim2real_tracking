"""Per-slot glimpse encoder + decoder.

Encoder: read a glimpse from the image at z_where, encode to a feature vector that is added
into z_what (dual-source head — see model/heads.py).

Decoder: z_what → small ConvT stack → patch with (appearance, mask_logit) channels.
The patch is later placed onto the canvas via stn_write.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from sim2real.model.nets import add_coords
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

    Uses CoordConv (coord channels concatenated before each ConvT) so the kernels can learn
    position-aware patterns — e.g. "Gaussian blob peaked at patch center", which the bare
    shift-equivariant convs cannot express directly.
    """

    glimpse_size: int = 16
    z_what_dim: int = 128
    channels: tuple[int, ...] = (64, 32)
    use_coord_conv: bool = True

    @nn.compact
    def __call__(self, z_what):
        base = 4
        x = nn.Dense(base * base * self.channels[0])(z_what)
        x = nn.gelu(x).reshape(base, base, self.channels[0])

        cur = base
        for c in self.channels[1:]:
            if self.use_coord_conv:
                x = add_coords(x)
            x = nn.ConvTranspose(c, (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.gelu(x)
            cur *= 2
        while cur < self.glimpse_size:
            if self.use_coord_conv:
                x = add_coords(x)
            x = nn.ConvTranspose(self.channels[-1], (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.gelu(x)
            cur *= 2
        x = x[: self.glimpse_size, : self.glimpse_size]
        if self.use_coord_conv:
            x = add_coords(x)
        x = nn.Conv(2, (1, 1))(x)
        appearance = nn.sigmoid(x[..., 0:1])              # (g, g, 1)
        mask_logit = x[..., 1:2]
        return appearance, mask_logit                     # both (g, g, 1)


class SegHead(nn.Module):
    """Per-slot segmentation head with spatial inductive bias.

    Mirrors `GlimpseDecoder`'s ConvT architecture but has its own parameters so it can specialize
    for binary cell shapes while the appearance decoder handles texture. Input is the slot's
    [z_what; z_where] concatenation, mapped to a (glimpse_size, glimpse_size, 1) mask logit patch.
    """

    glimpse_size: int = 16
    channels: tuple = (64, 32)        # ConvT channels (matches GlimpseDecoder.channels in spirit)
    base_res: int = 4

    @nn.compact
    def __call__(self, z_what, z_where):
        s = nn.sigmoid(z_where[..., 0:1])
        txy = jnp.tanh(z_where[..., 1:3])
        feat = jnp.concatenate([z_what, s, txy], axis=-1)
        c0 = self.channels[0]
        x = nn.Dense(self.base_res * self.base_res * c0)(feat)
        x = nn.gelu(x).reshape(self.base_res, self.base_res, c0)

        cur = self.base_res
        for c in self.channels[1:]:
            x = nn.ConvTranspose(c, (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.gelu(x)
            cur *= 2
        while cur < self.glimpse_size:
            x = nn.ConvTranspose(self.channels[-1], (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.gelu(x)
            cur *= 2
        x = x[: self.glimpse_size, : self.glimpse_size]
        # 1x1 conv to a single logit channel. Init bias negative so the initial sigmoid is small
        # (≈0.12) — prevents the "uniform 0.5 mask everywhere at step 0" failure.
        x = nn.Conv(
            1, (1, 1),
            bias_init=nn.initializers.constant(-2.0),
        )(x)
        return x                                                                      # (g, g, 1)
