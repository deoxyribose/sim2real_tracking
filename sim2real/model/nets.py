"""Small reusable network building blocks (linen)."""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jnp.ndarray


def zero_init(key, shape, dtype=jnp.float32):
    del key
    return jnp.zeros(shape, dtype=dtype)


def add_coords(x: Array) -> Array:
    """Concat normalized (y, x) coord channels to a (H, W, C) feature map.

    Output shape: (H, W, C + 2). Coord values in [-1, 1]. Lets following convs learn
    position-aware patterns (CoordConv — Liu et al. 2018, arXiv:1807.03247).
    """
    h, w, _ = x.shape
    yy = jnp.broadcast_to(jnp.linspace(-1.0, 1.0, h)[:, None, None], (h, w, 1))
    xx = jnp.broadcast_to(jnp.linspace(-1.0, 1.0, w)[None, :, None], (h, w, 1))
    return jnp.concatenate([x, yy, xx], axis=-1)


class MLP(nn.Module):
    """Simple MLP — `hidden` is a sequence of layer widths, the last entry is the output dim."""

    hidden: Sequence[int]
    activation: str = "gelu"
    final_zero_init: bool = False

    @nn.compact
    def __call__(self, x):
        act = {"gelu": nn.gelu, "relu": nn.relu, "tanh": jnp.tanh, "silu": nn.silu}[self.activation]
        for i, w in enumerate(self.hidden[:-1]):
            x = nn.Dense(w)(x)
            x = act(x)
        last_kernel_init = nn.initializers.zeros if self.final_zero_init else nn.initializers.lecun_normal()
        x = nn.Dense(self.hidden[-1], kernel_init=last_kernel_init)(x)
        return x


class LayerScaledResidual(nn.Module):
    """Residual block: x + scale * f(LN(x))."""

    inner: nn.Module
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        scale = self.param("scale", lambda key: jnp.array(self.init_scale))
        y = self.inner(nn.LayerNorm()(x), *args, **kwargs)
        return x + scale * y


def gru_step(h: Array, x: Array, *, hidden_dim: int, name: str = "gru") -> Array:
    """Run one step of a GRU cell. Stateful via flax — must be called inside a linen Module.

    Provided as a thin convenience wrapper around nn.GRUCell.
    """
    cell = nn.GRUCell(features=hidden_dim, name=name)
    new_h, _ = cell(h, x)
    return new_h


class MultiHeadSelfAttention(nn.Module):
    d_model: int
    n_heads: int = 8

    @nn.compact
    def __call__(self, x, mask=None):
        return nn.SelfAttention(num_heads=self.n_heads, qkv_features=self.d_model)(x, mask=mask)


class MultiHeadCrossAttention(nn.Module):
    d_model: int
    n_heads: int = 8

    @nn.compact
    def __call__(self, q, kv, mask=None):
        attn = nn.MultiHeadDotProductAttention(num_heads=self.n_heads, qkv_features=self.d_model)
        return attn(inputs_q=q, inputs_k=kv, inputs_v=kv, mask=mask)


class SlotCompetingCrossAttention(nn.Module):
    """Cross-attention with softmax OVER SLOTS instead of keys (Slot Attention style).

    Standard cross-attention: attn[i, j] = softmax_j(Q_i·K_j). Each slot i distributes attention
    over keys j; multiple slots can attend to the same key with no cost. This is why our current
    arch produces duplicate slots on the same object.

    Slot Attention: attn[i, j] = softmax_i(Q_i·K_j), then renormalize per-slot over keys. Each
    key j is (softly) assigned to exactly one slot i, so slots compete for image regions.

    Args:
      q:  (N, d_model)
      kv: (L, d_model)
    """
    d_model: int
    n_heads: int = 8

    @nn.compact
    def __call__(self, q, kv, mask=None):
        # (mask ignored — this attention is symmetric-competing and doesn't take a per-token mask.)
        d_head = self.d_model // self.n_heads
        Q = nn.Dense(self.d_model, name="q_proj")(q).reshape(-1, self.n_heads, d_head)   # (N, H, D)
        K = nn.Dense(self.d_model, name="k_proj")(kv).reshape(-1, self.n_heads, d_head)  # (L, H, D)
        V = nn.Dense(self.d_model, name="v_proj")(kv).reshape(-1, self.n_heads, d_head)  # (L, H, D)
        logits = jnp.einsum("nhd,lhd->hnl", Q, K) / jnp.sqrt(d_head)                     # (H, N, L)
        # Softmax over the SLOT axis (dim 1). Each key is softly claimed by one slot.
        attn = jax.nn.softmax(logits, axis=1)                                             # (H, N, L)
        # Renormalize each slot's attention so it sums to 1 across keys (Slot Attention convention).
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)                          # (H, N, L)
        out = jnp.einsum("hnl,lhd->nhd", attn, V).reshape(-1, self.d_model)              # (N, d)
        return nn.Dense(self.d_model, name="out_proj")(out)


class TransformerBlock(nn.Module):
    """Pre-norm decoder block: self-attn → cross-attn → FFN.

    When `slot_competing_cross=True`, the cross-attention softmax is over slots instead of keys,
    turning this into a Slot-Attention-like competition mechanism.
    """

    d_model: int
    n_heads: int = 8
    mlp_ratio: int = 4
    slot_competing_cross: bool = False

    @nn.compact
    def __call__(self, q, kv, cross_mask=None):
        # Self-attention
        h = nn.LayerNorm()(q)
        h = MultiHeadSelfAttention(self.d_model, self.n_heads)(h)
        q = q + h
        # Cross-attention (either standard or slot-competing)
        h = nn.LayerNorm()(q)
        if self.slot_competing_cross:
            h = SlotCompetingCrossAttention(self.d_model, self.n_heads)(h, kv, mask=cross_mask)
        else:
            h = MultiHeadCrossAttention(self.d_model, self.n_heads)(h, kv, mask=cross_mask)
        q = q + h
        # FFN
        h = nn.LayerNorm()(q)
        h = nn.Dense(self.d_model * self.mlp_ratio)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model)(h)
        return q + h
