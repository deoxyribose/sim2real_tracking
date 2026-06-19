"""Shared rendering helpers built on top of primitives + splines.

Every simulator funnels through these so that masks, composites, and noise share semantics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sim2real.sim.primitives import alpha_composite_with_bg, gaussian_blob, make_grid
from sim2real.sim.background import perlin_background

Array = jnp.ndarray


# --- z_where encoding helpers (5-dim affine) -------------------------------------------------

_S_CLIP = (1e-3, 0.97)        # keeps logit finite
_T_CLIP = 0.97                # keeps arctanh finite
_TH_CLIP = 0.97               # for theta / π


def pack_zwhere(
    sx: Array,
    sy: Array,
    theta: Array,
    tx: Array,
    ty: Array,
    scale_factor: float | tuple[float, float] = 1.0,
) -> Array:
    """Convert decoded values to the unconstrained raw form used by the model.

    `scale_factor` widens (sx, sy) before packing — useful so that the slot's z_where window
    covers the FULL mask extent (e.g., 3σ for a Gaussian) rather than just the visible center
    (~σ). Passing a tuple `(fx, fy)` lets `sx` and `sy` get different factors (e.g., elongated
    objects only need width padding, not length padding).

    All inputs broadcast; shapes preserved. Output last-dim is 5.
    """
    if isinstance(scale_factor, tuple):
        fx, fy = scale_factor
    else:
        fx = fy = scale_factor
    sx_raw = jax.scipy.special.logit(jnp.clip(sx * fx, _S_CLIP[0], _S_CLIP[1]))
    sy_raw = jax.scipy.special.logit(jnp.clip(sy * fy, _S_CLIP[0], _S_CLIP[1]))
    theta_raw = jnp.arctanh(jnp.clip(theta / jnp.pi, -_TH_CLIP, _TH_CLIP))
    tx_raw = jnp.arctanh(jnp.clip(tx, -_T_CLIP, _T_CLIP))
    ty_raw = jnp.arctanh(jnp.clip(ty, -_T_CLIP, _T_CLIP))
    return jnp.stack([sx_raw, sy_raw, theta_raw, tx_raw, ty_raw], axis=-1)


def oriented_extent(points: Array) -> tuple[Array, Array, Array, Array]:
    """PCA-based oriented bounding box for a (P, 2) point cloud.

    Returns (centroid_xy(2,), theta_scalar, sx_half_scalar, sy_half_scalar) where theta is the
    angle of the principal axis in radians (canonicalized so the principal direction has a
    non-negative x component) and sx_half / sy_half are the maximum |projection| of the
    centered points onto the principal / perpendicular axes.
    """
    centroid = jnp.mean(points, axis=0)                                                # (2,)
    centered = points - centroid                                                       # (P, 2)
    cov = (centered.T @ centered) / centered.shape[0]                                  # (2, 2)
    eigvals, eigvecs = jnp.linalg.eigh(cov)                                            # ascending
    v1 = eigvecs[:, 1]                                                                  # principal
    # Canonicalize: ensure v1[0] >= 0 (flip otherwise) — kills the 180° sign ambiguity.
    sign = jnp.where(v1[0] >= 0, 1.0, -1.0)
    v1 = v1 * sign
    v2 = jnp.array([-v1[1], v1[0]])                                                    # 90° rot
    theta = jnp.arctan2(v1[1], v1[0])                                                  # in [-π/2, π/2]
    proj1 = centered @ v1
    proj2 = centered @ v2
    sx_half = jnp.maximum(jnp.max(jnp.abs(proj1)), 1e-3)
    sy_half = jnp.maximum(jnp.max(jnp.abs(proj2)), 1e-3)
    return centroid, theta, sx_half, sy_half


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
