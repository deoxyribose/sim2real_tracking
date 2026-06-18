"""Affine spatial transformer (read/write) with 5-dim z_where.

Conventions:
  z_where = (sx_raw, sy_raw, theta_raw, tx_raw, ty_raw)
  sx    = sigmoid(sx_raw)     ∈ (0, 1)  — half-extent along the patch's x axis
  sy    = sigmoid(sy_raw)     ∈ (0, 1)  — half-extent along the patch's y axis
  theta = π · tanh(theta_raw) ∈ (-π, π) — rotation of patch frame relative to canvas
  tx    = tanh(tx_raw)        ∈ (-1, 1) — center in canvas-x, normalized
  ty    = tanh(ty_raw)        ∈ (-1, 1) — center in canvas-y, normalized

Forward affine (patch grid → canvas):
    image_xy = (tx, ty) + R(θ) · diag(sx, sy) · patch_xy
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def _decode_zwhere(z_where: Array):
    sx = jax.nn.sigmoid(z_where[..., 0])
    sy = jax.nn.sigmoid(z_where[..., 1])
    theta = jnp.pi * jnp.tanh(z_where[..., 2])
    tx = jnp.tanh(z_where[..., 3])
    ty = jnp.tanh(z_where[..., 4])
    return sx, sy, theta, tx, ty


def stn_read(image: Array, z_where: Array, glimpse_size: int) -> Array:
    """Read a (glimpse_size, glimpse_size, C) glimpse from `image` at the affine `z_where`."""
    H, W, C = image.shape
    sx, sy, theta, tx, ty = _decode_zwhere(z_where)
    cos = jnp.cos(theta)
    sin = jnp.sin(theta)

    # Patch grid in [-1, 1]² (axis-aligned in patch frame).
    grid_n = jnp.linspace(-1.0, 1.0, glimpse_size)
    gy, gx = jnp.meshgrid(grid_n, grid_n, indexing="ij")               # (g, g)

    # Apply affine: scale then rotate then translate (in normalized canvas coords).
    sxgx = sx * gx
    sygy = sy * gy
    src_x_norm = tx + cos * sxgx - sin * sygy
    src_y_norm = ty + sin * sxgx + cos * sygy

    # Convert to pixel coords.
    src_x = (src_x_norm + 1.0) * 0.5 * (W - 1)
    src_y = (src_y_norm + 1.0) * 0.5 * (H - 1)

    def sample_channel(img_c):
        return jax.scipy.ndimage.map_coordinates(img_c, [src_y, src_x], order=1, mode="constant", cval=0.0)

    return jnp.stack([sample_channel(image[..., c]) for c in range(C)], axis=-1)


def stn_write(patch: Array, z_where: Array, out_size: int) -> Array:
    """Place a (gh, gw, C) `patch` onto an (out_size, out_size, C) canvas at the affine `z_where`.

    Inverse-warp: for each canvas pixel, compute its location in patch coords and bilinear-sample.
    """
    gh, gw, C = patch.shape
    sx, sy, theta, tx, ty = _decode_zwhere(z_where)
    cos = jnp.cos(theta)
    sin = jnp.sin(theta)

    # Output canvas grid in [-1, 1]².
    grid_n = jnp.linspace(-1.0, 1.0, out_size)
    oy, ox = jnp.meshgrid(grid_n, grid_n, indexing="ij")               # (out, out)

    # Inverse affine: (canvas - translation) → R⁻¹(θ) → diag(1/sx, 1/sy)
    dx = ox - tx
    dy = oy - ty
    # R⁻¹(θ) = R(-θ): (cos -sin; sin cos) inverted.
    rx = cos * dx + sin * dy
    ry = -sin * dx + cos * dy
    px_n = rx / (sx + 1e-8)
    py_n = ry / (sy + 1e-8)
    inside = (jnp.abs(px_n) <= 1.0) & (jnp.abs(py_n) <= 1.0)

    # Convert to patch pixel coords.
    py = (py_n + 1.0) * 0.5 * (gh - 1)
    px = (px_n + 1.0) * 0.5 * (gw - 1)

    def sample_channel(p_c):
        v = jax.scipy.ndimage.map_coordinates(p_c, [py, px], order=1, mode="constant", cval=0.0)
        return jnp.where(inside, v, 0.0)

    return jnp.stack([sample_channel(patch[..., c]) for c in range(C)], axis=-1)
