"""Sinusoidal 2D positional encoding for image features."""

from __future__ import annotations

import jax.numpy as jnp


def sinusoidal_2d(h: int, w: int, dim: int) -> jnp.ndarray:
    """Return a (h, w, dim) 2D sinusoidal positional encoding. `dim` must be a multiple of 4."""
    assert dim % 4 == 0, f"dim={dim} must be multiple of 4 for 2D sinusoidal PE"
    dq = dim // 2  # half for x, half for y
    div_term = jnp.exp(-jnp.log(10000.0) * jnp.arange(0, dq, 2) / dq)            # (dq/2,)

    py = jnp.arange(h, dtype=jnp.float32)[:, None]                              # (h, 1)
    px = jnp.arange(w, dtype=jnp.float32)[:, None]                              # (w, 1)

    pe_y = jnp.concatenate(
        [jnp.sin(py * div_term), jnp.cos(py * div_term)], axis=-1
    )                                                                            # (h, dq)
    pe_x = jnp.concatenate(
        [jnp.sin(px * div_term), jnp.cos(px * div_term)], axis=-1
    )                                                                            # (w, dq)

    pe = jnp.concatenate(
        [
            jnp.broadcast_to(pe_y[:, None, :], (h, w, dq)),
            jnp.broadcast_to(pe_x[None, :, :], (h, w, dq)),
        ],
        axis=-1,
    )                                                                            # (h, w, dim)
    return pe
