"""Reconstruction loss: MSE on the clipped composite (v20 lesson)."""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray


def recon_mse(composite: Array, target: Array, weight: Array | None = None) -> Array:
    """MSE between (T, H, W, C) composite and target.

    If `weight` is provided it must broadcast against `(composite - target)**2` (typically
    shape (T, H, W, 1) or (T, H, W)). Uses a weighted mean: sum(w * err) / sum(w).
    """
    sq = (composite - target) ** 2
    if weight is None:
        return jnp.mean(sq)
    if weight.ndim == sq.ndim - 1:
        weight = weight[..., None]
    return jnp.sum(weight * sq) / (jnp.sum(weight) * sq.shape[-1] + 1e-6)
