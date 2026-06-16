"""Shared rendering helpers built on top of primitives + splines.

Every simulator funnels through these so that masks, composites, and noise share semantics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sim2real.sim.primitives import alpha_composite_with_bg, gaussian_blob, make_grid
from sim2real.sim.background import perlin_background

Array = jnp.ndarray


def render_object_appearance(soft_mask: Array, color: Array) -> Array:
    """Multiply a (H,W) soft mask by a (C,) color vector → (H,W,C) appearance layer."""
    return soft_mask[..., None] * color[None, None, :]


def composite_video_frame(
    masks: Array, appearances: Array, bg: Array, z_pres: Array
) -> Array:
    """Compose one frame from per-slot masks/appearances over a background, gated by z_pres.

    Args:
      masks:       (N, H, W) in [0,1].
      appearances: (N, H, W, C).
      bg:          (H, W, C).
      z_pres:      (N,) in [0,1].
    """
    gated_masks = masks * z_pres[:, None, None]
    return alpha_composite_with_bg(appearances, gated_masks, bg)


def add_observation_noise(key: jax.Array, frame: Array, sigma: float) -> Array:
    """Gaussian observation noise + clip to [0,1]."""
    noise = sigma * jax.random.normal(key, frame.shape)
    return jnp.clip(frame + noise, 0.0, 1.0)


def gaussian_blob_mask(res: int, cx: float, cy: float, sigma: float) -> Array:
    """Soft circular mask in [-1,1]² coordinates, output shape (res, res)."""
    xx, yy = make_grid(res)
    return gaussian_blob(xx, yy, cx, cy, sigma)


def perlin_grayscale_bg(key: jax.Array, res: int, channels: int = 1) -> Array:
    """Generate a grayscale background with shape (res, res, channels)."""
    g = perlin_background(key, res)
    if channels == 1:
        return g[..., None]
    return jnp.broadcast_to(g[..., None], (res, res, channels))
