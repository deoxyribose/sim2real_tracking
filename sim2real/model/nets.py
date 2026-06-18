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


class TransformerBlock(nn.Module):
    """Pre-norm decoder block: self-attn → cross-attn → FFN."""

    d_model: int
    n_heads: int = 8
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, q, kv, cross_mask=None):
        # Self-attention
        h = nn.LayerNorm()(q)
        h = MultiHeadSelfAttention(self.d_model, self.n_heads)(h)
        q = q + h
        # Cross-attention
        h = nn.LayerNorm()(q)
        h = MultiHeadCrossAttention(self.d_model, self.n_heads)(h, kv, mask=cross_mask)
        q = q + h
        # FFN
        h = nn.LayerNorm()(q)
        h = nn.Dense(self.d_model * self.mlp_ratio)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model)(h)
        return q + h
