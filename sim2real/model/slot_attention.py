"""Proper iterative Slot Attention (Locatello et al. 2020), SAVi-style recurrent.

Differences vs our current DETR-style slot_transformer:
  - Softmax over the SLOT axis (not key axis) — slots compete for image regions per patch
  - Value normalization per slot (each slot's attention sums to 1 across keys)
  - Iterative refinement (n_iters passes with the SAME learned weights, not n_layers of different weights)
  - GRU update between iterations (not residual add)

Signature is a drop-in replacement for SlotTransformer's __call__ output tuple so the rest of
the model doesn't change.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from sim2real.model.posenc import sinusoidal_2d

Array = jnp.ndarray


class SlotAttentionRefiner(nn.Module):
    """One `n_iters`-step Slot Attention refinement. Weights shared across iterations.

    Given image features (L, D) and initial slots (N, D), returns refined slots (N, D).
    """

    d_model: int
    n_iters: int = 3

    @nn.compact
    def __call__(self, features, slots):
        # Learned projections applied once at every iteration.
        norm_slots = nn.LayerNorm(name="slot_norm")
        norm_feats = nn.LayerNorm(name="feat_norm")
        norm_pre_mlp = nn.LayerNorm(name="pre_mlp_norm")
        q_proj = nn.Dense(self.d_model, use_bias=False, name="q_proj")
        k_proj = nn.Dense(self.d_model, use_bias=False, name="k_proj")
        v_proj = nn.Dense(self.d_model, use_bias=False, name="v_proj")
        gru = nn.GRUCell(features=self.d_model, name="gru")
        mlp_hidden = self.d_model * 4
        mlp_1 = nn.Dense(mlp_hidden, name="mlp_1")
        mlp_2 = nn.Dense(self.d_model, name="mlp_2")

        # Normalize features once; projections re-run per iter.
        feats_n = norm_feats(features)                                                  # (L, D)
        K = k_proj(feats_n)                                                             # (L, D)
        V = v_proj(feats_n)                                                             # (L, D)
        scale = self.d_model ** -0.5

        for _ in range(self.n_iters):
            slots_n = norm_slots(slots)                                                 # (N, D)
            Q = q_proj(slots_n)                                                         # (N, D)
            logits = (Q @ K.T) * scale                                                  # (N, L)
            # Softmax OVER SLOTS (dim 0) — each key is (softly) claimed by one slot.
            attn = jax.nn.softmax(logits, axis=0)                                       # (N, L)
            # Renormalize each slot's attention so it sums to 1 across keys.
            attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)                     # (N, L)
            updates = attn @ V                                                          # (N, D)
            # GRU update: current slot state ← f(slot, update).
            slots, _ = gru(updates, slots)
            # Residual MLP.
            slots = slots + mlp_2(nn.gelu(mlp_1(norm_pre_mlp(slots))))                  # (N, D)
        return slots


class SlotAttentionStack(nn.Module):
    """SAVi-style stack: sinusoidal PE → Slot Attention refinement.

    Signature matches SlotTransformer.__call__ for drop-in substitution:
    returns (q_prop, q_disc, memory, q_prop_layers, q_disc_layers). Since Slot Attention
    has no propagate/discover split, both q_prop and q_disc are the same refined slots.
    """

    n_max: int
    d_model: int
    n_iters: int = 3
    _dummy_n_layers: int = 3   # only used to keep len(q_disc_layers) matching the DETR-aux path

    @nn.compact
    def __call__(self, feat_grid, slot_h, residual_mask_pixel=None):
        h, w, d = feat_grid.shape
        pe = sinusoidal_2d(h, w, d)
        features = (feat_grid + pe).reshape(h * w, d)                                   # (L, D)

        # Slot init: previous-frame slot state (carried from earlier scan step) + a learned bias
        # that breaks symmetry at t=0 when slot_h is zeros. Locatello samples from N(mu, sigma);
        # we use a learned mean directly — trivially identical in effect and matches our carry pattern.
        slot_bias = self.param(
            "slot_init_bias",
            nn.initializers.normal(stddev=0.2),
            (self.n_max, self.d_model),
        )
        slots = slot_h + slot_bias                                                      # (N, D)

        slots = SlotAttentionRefiner(self.d_model, self.n_iters)(features, slots)       # (N, D)

        # Same output for prop/disc; layer lists have _dummy_n_layers entries all equal
        # to the final slots (aux DETR supervision loss will treat them all as the same).
        layers = [slots] * self._dummy_n_layers
        return slots, slots, features, layers, layers
