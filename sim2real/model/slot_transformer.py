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
from sim2real.model.slot_attention import SlotAttentionStack

Array = jnp.ndarray


class SlotTokens(nn.Module):
    """Per-slot query embeddings. Two modes:

    - Default (`anchor=False`): learnable per-slot vectors, N × d_model. Standard DETR.
    - `anchor=True`: fixed grid anchor positions in [-1,1]², encoded via sinusoidal features
      and projected to d_model via a learnable Dense. Each slot's identity is tied to a
      specific spatial location, so its transformer output depends coherently on scene content
      near that anchor. Attacks the "learnable slot queries get pulled to the sim centroid
      under permutation-averaged gradients" failure mode.
    """

    n_max: int
    d_model: int
    anchor: bool = False
    n_freqs: int = 8   # number of sinusoidal frequencies per axis (only used when anchor=True)

    @nn.compact
    def __call__(self):
        if not self.anchor:
            return self.param(
                "slot_emb",
                nn.initializers.normal(stddev=0.2),
                (self.n_max, self.d_model),
            )
        # Anchor grid in [-0.8, 0.8]² (leave margin from edges).
        import math
        n = self.n_max
        rows = int(round(math.sqrt(n)))
        cols = int(math.ceil(n / rows))
        ys = jnp.linspace(-0.8, 0.8, rows)
        xs = jnp.linspace(-0.8, 0.8, cols)
        yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
        pos = jnp.stack([yy.flatten(), xx.flatten()], axis=-1)[:n]         # (n, 2)
        # Multi-frequency sinusoidal encoding.
        freqs = 2.0 ** jnp.arange(self.n_freqs, dtype=jnp.float32) * jnp.pi
        phases = pos[:, :, None] * freqs[None, None, :]                    # (n, 2, n_freqs)
        feats = jnp.concatenate(
            [jnp.sin(phases), jnp.cos(phases)], axis=-1
        ).reshape(n, -1)                                                    # (n, 4 * n_freqs)
        return nn.Dense(self.d_model, name="anchor_proj")(feats)           # (n, d_model)


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
    slot_competing_cross: bool = False
    anchor_slots: bool = False
    # When True, use proper iterative Slot Attention (Locatello 2020, SAVi-style) instead of
    # the DETR-style transformer stack. Overrides slot_competing_cross and n_layers-per-block
    # behavior. n_layers is reinterpreted as the number of Slot Attention iterations.
    use_slot_attention: bool = False

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
        # Fast path: proper iterative Slot Attention (Locatello 2020 style).
        if self.use_slot_attention:
            return SlotAttentionStack(
                n_max=self.n_max, d_model=self.d_model, n_iters=self.n_layers,
            )(feat_grid, slot_h, residual_mask_pixel)

        h, w, d = feat_grid.shape
        pe = sinusoidal_2d(h, w, d)
        memory = (feat_grid + pe).reshape(h * w, d)                                # (L, d)

        slot_emb = SlotTokens(self.n_max, self.d_model, anchor=self.anchor_slots)()
        q = slot_emb + slot_h                                                       # (N, d)

        q_prop = q
        q_prop_layers = []
        for _ in range(self.n_layers):
            q_prop = TransformerBlock(self.d_model, self.n_heads, slot_competing_cross=self.slot_competing_cross)(q_prop, memory)
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
            q_disc = TransformerBlock(self.d_model, self.n_heads, slot_competing_cross=self.slot_competing_cross)(q_disc, disc_memory, cross_mask)
            q_disc_layers.append(q_disc)

        return q_prop, q_disc, memory, q_prop_layers, q_disc_layers
