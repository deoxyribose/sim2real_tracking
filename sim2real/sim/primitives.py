"""Differentiable rendering primitives shared by all simulators.

All functions are pure JAX and `vmap`-friendly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def make_grid(res: int) -> tuple[Array, Array]:
    """Return (xx, yy) pixel-coordinate grids of shape (res, res), centered at 0 with range [-1, 1]."""
    lin = jnp.linspace(-1.0, 1.0, res)
    xx, yy = jnp.meshgrid(lin, lin, indexing="xy")
    return xx, yy


def gaussian_blob(xx: Array, yy: Array, cx: float, cy: float, sigma: float) -> Array:
    """Isotropic Gaussian blob evaluated on the (xx, yy) grid. Output in [0, 1]."""
    return jnp.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / (sigma ** 2 + 1e-8))


def soft_mask_from_sdf(sdf: Array, sharpness: float = 8.0) -> Array:
    """Convert a signed distance field (positive = inside) to a [0,1] soft mask."""
    return jax.nn.sigmoid(sharpness * sdf)


def alpha_composite_over(layers: Array, masks: Array, eps: float = 1e-6) -> Array:
    """Composite N layers with their masks using weighted-average (matches the model decoder).

    Args:
      layers: (N, H, W, C) appearance per slot, in [0,1].
      masks:  (N, H, W)    occupancy per slot, in [0,1] (often = z_pres × soft_mask).

    Returns:
      (H, W, C) — `sum_n layer_n * mask_n / (eps + sum_n mask_n)`, clipped to [0,1].
    """
    w = masks[..., None]  # (N,H,W,1)
    num = jnp.sum(layers * w, axis=0)
    den = jnp.sum(w, axis=0) + eps
    out = num / den
    return jnp.clip(out, 0.0, 1.0)


def alpha_composite_with_bg(layers: Array, masks: Array, bg: Array) -> Array:
    """Composite N layers over a background image using sequential 'over' blending.

    Args:
      layers: (N, H, W, C)
      masks:  (N, H, W)
      bg:     (H, W, C)

    Returns:
      (H, W, C) clipped to [0,1].
    """
    # Sort-free associative-ish composite: weighted-average over slots, then over background.
    fg_num = jnp.sum(layers * masks[..., None], axis=0)
    fg_w = jnp.sum(masks, axis=0)[..., None]  # (H,W,1)
    fg = fg_num / (fg_w + 1e-6)
    alpha = jnp.clip(fg_w, 0.0, 1.0)
    out = alpha * fg + (1.0 - alpha) * bg
    return jnp.clip(out, 0.0, 1.0)


def stamp_at(patch: Array, center: tuple[float, float], scale: float, out_res: int) -> Array:
    """Place a small `patch` of shape (gh, gw, C) onto an (out_res, out_res, C) canvas at `center`
    with `scale` applied (1.0 = native patch size). Uses bilinear scale_and_translate.

    Coordinates: center in normalized [-1, 1] space (matches `make_grid`).
    """
    gh, gw, c = patch.shape
    cx, cy = center
    # Convert normalized -> pixel
    px = (cx + 1.0) * 0.5 * out_res
    py = (cy + 1.0) * 0.5 * out_res
    target_size = jnp.round(scale * gh).astype(jnp.int32)
    # jax.image.scale_and_translate works in pixel coords:
    #   output[i,j] = patch sampled at scaled & translated location.
    scaled = jax.image.scale_and_translate(
        patch,
        shape=(out_res, out_res, c),
        spatial_dims=(0, 1),
        scale=jnp.array([scale, scale]),
        translation=jnp.array([py - 0.5 * scale * gh, px - 0.5 * scale * gw]),
        method="linear",
        antialias=True,
    )
    del target_size
    return scaled
