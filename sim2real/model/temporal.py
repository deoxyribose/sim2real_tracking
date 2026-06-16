"""Per-slot 1-step GRU state update.

The GRU sees `[refined_q ; z_where ; z_what ; z_pres]` and produces the next slot hidden state.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

Array = jnp.ndarray


class SlotGRU(nn.Module):
    d_model: int = 256

    @nn.compact
    def __call__(self, h_prev: Array, x: Array) -> Array:
        """h_prev: (d_model,); x: (input_dim,) — concatenation of refined_q and current latents."""
        cell = nn.GRUCell(features=self.d_model)
        new_h, _ = cell(h_prev, x)
        return new_h
