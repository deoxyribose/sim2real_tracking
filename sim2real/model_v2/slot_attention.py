"""Vanilla slot attention (Locatello 2020).

K learnable queries iterate T times over a feature map. At each iteration:
  - K, V from features (input-dependent)
  - Q from current slot state
  - attention softmax OVER SLOTS (not spatial), which is the load-bearing choice
  - weighted-average pooling of values per slot
  - GRU update + MLP residual per slot

No pose factorization here — that's ISA's contribution. For MVP we use vanilla
Locatello attention: simpler, standard, and enough to get end-to-end training working.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class SlotAttention(nn.Module):
    n_slots: int = 8
    n_iters: int = 3
    d_slot: int = 128
    d_ff: int = 256
    eps: float = 1e-8

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        """features: (B, L, D). Returns slots: (B, n_slots, d_slot)."""
        B, L, D = features.shape
        # Per-slot LEARNED initial vectors (not shared across slots) — better symmetry
        # breaking than shared-mu + gaussian noise, which tends to collapse when the
        # noise gets averaged out over training.
        slot_init = self.param("slot_init", nn.initializers.normal(stddev=0.5),
                                (self.n_slots, self.d_slot))
        # Small per-batch noise on top, so each batch element still has diversity.
        key = self.make_rng("slots")
        noise = jax.random.normal(key, (B, self.n_slots, self.d_slot)) * 0.1
        slots = slot_init[None] + noise

        # Linear projections
        to_k = nn.Dense(self.d_slot, use_bias=False, name="to_k")
        to_v = nn.Dense(self.d_slot, use_bias=False, name="to_v")
        to_q = nn.Dense(self.d_slot, use_bias=False, name="to_q")

        norm_in = nn.LayerNorm(name="norm_in")
        norm_slot = nn.LayerNorm(name="norm_slot")
        norm_mlp = nn.LayerNorm(name="norm_mlp")

        gru = nn.GRUCell(self.d_slot, name="gru")

        mlp = nn.Sequential([
            nn.Dense(self.d_ff), jax.nn.gelu, nn.Dense(self.d_slot),
        ])

        f_norm = norm_in(features)
        K = to_k(f_norm)  # (B, L, d)
        V = to_v(f_norm)  # (B, L, d)

        for _ in range(self.n_iters):
            s_norm = norm_slot(slots)
            Q = to_q(s_norm)  # (B, n_slots, d)
            # softmax over slots (per pixel, which slot claims it)
            logits = jnp.einsum("bsd,bld->bsl", Q, K) / jnp.sqrt(self.d_slot)  # (B, n_slots, L)
            attn = jax.nn.softmax(logits, axis=1)                              # softmax OVER SLOTS
            # normalize per slot
            attn = attn + self.eps
            attn = attn / attn.sum(axis=-1, keepdims=True)                     # normalize over spatial
            updates = jnp.einsum("bsl,bld->bsd", attn, V)                      # (B, n_slots, d)

            # GRU update, then MLP residual
            # GRUCell expects (carry, x) → (new_carry, y). We use per-slot GRU.
            slots_new, _ = jax.vmap(jax.vmap(gru))(slots, updates)
            slots = slots_new + mlp(norm_mlp(slots_new))

        return slots
