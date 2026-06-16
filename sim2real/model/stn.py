"""Spatial transformer (read/write) with z_where = (s_raw, tx_raw, ty_raw) in tanh-space.

Conventions:
  s   = sigmoid(s_raw)  ∈ (0, 1)  — half-extent (1.0 ≈ filling the canvas).
  tx  = tanh(tx_raw)    ∈ (-1, 1) — normalized center in x (image is [-1, 1]² space).
  ty  = tanh(ty_raw)    ∈ (-1, 1) — normalized center in y.

Both `stn_read` and `stn_write` use bilinear interpolation via `jax.scipy.ndimage.map_coordinates`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def _decode_zwhere(z_where: Array) -> tuple[Array, Array, Array]:
    s_raw, tx_raw, ty_raw = z_where[..., 0], z_where[..., 1], z_where[..., 2]
    return jax.nn.sigmoid(s_raw), jnp.tanh(tx_raw), jnp.tanh(ty_raw)


def stn_read(image: Array, z_where: Array, glimpse_size: int) -> Array:
    """Read a (glimpse_size, glimpse_size, C) glimpse from `image` at `z_where`.

    Args:
      image: (H, W, C).
      z_where: (3,).
      glimpse_size: side length of the glimpse.

    Returns:
      glimpse: (glimpse_size, glimpse_size, C).
    """
    H, W, C = image.shape
    s, tx, ty = _decode_zwhere(z_where)
    half = s  # half-extent in normalized [-1, 1] space
    # Build sampling coords in normalized space.
    grid_n = jnp.linspace(-1.0, 1.0, glimpse_size)
    gy, gx = jnp.meshgrid(grid_n, grid_n, indexing="ij")           # (g, g)
    # Map normalized glimpse coords (in [-1,1]) into image normalized coords centered at (tx, ty).
    src_x_norm = tx + half * gx                                     # (g, g)
    src_y_norm = ty + half * gy
    # Convert to pixel coords.
    src_x = (src_x_norm + 1.0) * 0.5 * (W - 1)
    src_y = (src_y_norm + 1.0) * 0.5 * (H - 1)

    def sample_channel(img_c):
        return jax.scipy.ndimage.map_coordinates(img_c, [src_y, src_x], order=1, mode="constant", cval=0.0)

    return jnp.stack([sample_channel(image[..., c]) for c in range(C)], axis=-1)


def stn_write(patch: Array, z_where: Array, out_size: int) -> Array:
    """Place a (gh, gw, C) `patch` onto an (out_size, out_size, C) canvas at `z_where`.

    Inverse-warp: for each output pixel (in canvas), compute the corresponding patch coord and
    sample. Pixels outside the patch get 0.
    """
    gh, gw, C = patch.shape
    s, tx, ty = _decode_zwhere(z_where)
    half = s

    # Output canvas normalized grid.
    grid_n = jnp.linspace(-1.0, 1.0, out_size)
    oy, ox = jnp.meshgrid(grid_n, grid_n, indexing="ij")           # (out, out)
    # Map canvas coords back to patch coords.
    # Canvas normalized (oy, ox) -> patch normalized via inverse of (ty + half * py = oy).
    py_n = (oy - ty) / (half + 1e-8)
    px_n = (ox - tx) / (half + 1e-8)
    # Convert to patch pixel coords.
    py = (py_n + 1.0) * 0.5 * (gh - 1)
    px = (px_n + 1.0) * 0.5 * (gw - 1)

    # mask pixels outside patch [-1, 1]
    inside = (jnp.abs(py_n) <= 1.0) & (jnp.abs(px_n) <= 1.0)

    def sample_channel(p_c):
        v = jax.scipy.ndimage.map_coordinates(p_c, [py, px], order=1, mode="constant", cval=0.0)
        return jnp.where(inside, v, 0.0)

    return jnp.stack([sample_channel(patch[..., c]) for c in range(C)], axis=-1)
