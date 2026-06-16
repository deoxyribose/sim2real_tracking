"""Reconstruction loss: MSE on the clipped composite (v20 lesson)."""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray


def recon_mse(composite: Array, target: Array) -> Array:
    """MSE between (T, H, W, C) composite and target. Composite is already clipped in the model."""
    return jnp.mean((composite - target) ** 2)
