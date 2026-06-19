"""DETR-style slot transformer with propagate-and-discover passes.

Inputs:
  feat_grid: (H', W', d_model)  encoder features for one frame.
  slot_h:    (N, d_model)        slot hidden states (carried from previous frame; zero on frame 0).
  z_pres_prev: (N,)               previous-frame z_pres (informs which slots are dormant for discovery).

Outputs:
  q_prop: (N, d_model)            refined slot tokens after propagation.
  q_disc: (N, d_model)            refined slot tokens after discovery (separate cross-attention).
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from sim2real.model.nets import TransformerBlock
from sim2real.model.posenc import sinusoidal_2d

Array = jnp.ndarray


class SlotTokens(nn.Module):
    """Learnable per-slot query embeddings."""

    n_max: int
    d_model: int

    @nn.compact
    def __call__(self):
        # Returns the static slot embedding table.
        emb = self.param(
            "slot_emb",
            nn.initializers.normal(stddev=0.02),
            (self.n_max, self.d_model),
        )
        return emb


class SlotTransformer(nn.Module):
    """Two-pass slot decoder: propagate then discover.

    For propagate, the slots cross-attend to the full encoder grid.
    For discover, the same slots cross-attend to a *masked* encoder grid where alive slots'
    explained regions have been suppressed (key-side mask provided by caller; we just pass it on).
    """

    n_max: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 3

    @nn.compact
    def __call__(
        self,
        feat_grid: Array,
        slot_h: Array,
        residual_mask_pixel: Array | None = None,
    ):
        """
        Returns:
          q_prop:        (N, d_model)
          q_disc:        (N, d_model)
          memory:        (L, d_model) flattened feature tokens
          q_prop_layers: list[(N, d_model)] of length n_layers — per-layer q from propagate pass
          q_disc_layers: list[(N, d_model)] of length n_layers — per-layer q from discover pass
                         (the last entry is q_disc itself; used for DETR-style deep supervision)
        """
        h, w, d = feat_grid.shape
        pe = sinusoidal_2d(h, w, d)
        memory = (feat_grid + pe).reshape(h * w, d)                                # (L, d)

        slot_emb = SlotTokens(self.n_max, self.d_model)()
        q = slot_emb + slot_h                                                       # (N, d)

        q_prop = q
        q_prop_layers = []
        for _ in range(self.n_layers):
            q_prop = TransformerBlock(self.d_model, self.n_heads)(q_prop, memory)
            q_prop_layers.append(q_prop)

        if residual_mask_pixel is None:
            disc_memory = memory
            cross_mask = None
        else:
            rm = residual_mask_pixel.reshape(h * w, 1)
            disc_memory = memory * rm
            cross_mask = None

        q_disc = q_prop
        q_disc_layers = []
        for _ in range(self.n_layers):
            q_disc = TransformerBlock(self.d_model, self.n_heads)(q_disc, disc_memory, cross_mask)
            q_disc_layers.append(q_disc)

        return q_prop, q_disc, memory, q_prop_layers, q_disc_layers
