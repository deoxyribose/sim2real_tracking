"""Per-video background renderer.

Renders a full-frame (H, W, 1) appearance from z_style. Composited UNDER the active slots so
the model has an actual background to compare against, instead of falling back to all-zero in
unexplained regions. CoordConv before each ConvT.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from sim2real.model.nets import add_coords

Array = jnp.ndarray


class BackgroundRenderer(nn.Module):
    out_res: int = 128
    base_res: int = 8
    channels: tuple = (64, 32, 16)
    z_style_dim: int = 16
    use_coord_conv: bool = True

    @nn.compact
    def __call__(self, z_style):
        c0 = self.channels[0]
        x = nn.Dense(self.base_res * self.base_res * c0)(z_style)
        x = nn.gelu(x).reshape(self.base_res, self.base_res, c0)

        cur = self.base_res
        for c in self.channels[1:]:
            if self.use_coord_conv:
                x = add_coords(x)
            x = nn.ConvTranspose(c, (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.gelu(x)
            cur *= 2
        while cur < self.out_res:
            if self.use_coord_conv:
                x = add_coords(x)
            x = nn.ConvTranspose(self.channels[-1], (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.gelu(x)
            cur *= 2
        x = x[: self.out_res, : self.out_res]
        if self.use_coord_conv:
            x = add_coords(x)
        x = nn.Conv(1, (1, 1))(x)
        return nn.sigmoid(x)                                                          # (H, W, 1)
