"""Per-video background renderer.

Renders a full-frame (H, W, 1) appearance from z_style. Composited UNDER the active slots so
the model has an actual background to compare against, instead of falling back to all-zero in
unexplained regions.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

Array = jnp.ndarray


class BackgroundRenderer(nn.Module):
    """z_style → low-res feature map → ConvT stack → (out_res, out_res, 1) grayscale background."""

    out_res: int = 128
    base_res: int = 8                # initial spatial size after the MLP
    channels: tuple = (64, 32, 16)   # ConvT channel sequence
    z_style_dim: int = 16

    @nn.compact
    def __call__(self, z_style):
        # z_style: (Zs,) → flat feature (base_res * base_res * c0)
        c0 = self.channels[0]
        x = nn.Dense(self.base_res * self.base_res * c0)(z_style)
        x = nn.gelu(x).reshape(self.base_res, self.base_res, c0)

        cur = self.base_res
        for c in self.channels[1:]:
            x = nn.ConvTranspose(c, (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.gelu(x)
            cur *= 2
        # Keep upsampling until cur >= out_res
        while cur < self.out_res:
            x = nn.ConvTranspose(self.channels[-1], (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.gelu(x)
            cur *= 2
        # Crop / resize to exact out_res
        x = x[: self.out_res, : self.out_res]
        # Final 1x1 to a single grayscale channel; sigmoid for [0,1].
        x = nn.Conv(1, (1, 1))(x)
        return nn.sigmoid(x)                                                          # (H, W, 1)
