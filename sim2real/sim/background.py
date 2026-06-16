"""Background generation: Perlin-style noise from a random gradient grid.

We sample one random gradient angle per lattice cell using a `jax.random` key, then
bilinearly interpolate dot products with smooth fade — a value-noise / Perlin hybrid.
This is fully traceable (no host-side seeds, no integer overflow), unlike the original
algae implementation which relied on bit-mixed hash arithmetic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def _fade(t: Array) -> Array:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: Array, b: Array, t: Array) -> Array:
    return a + t * (b - a)


def perlin_noise(key: jax.Array, res: int, steps: int) -> Array:
    """Perlin-like noise on a (res, res) grid.

    `steps` lattice cells span the image (more = finer texture).
    Output: (res, res), approximately in [-1, 1].
    """
    cells = int(steps) + 1
    # One random gradient angle per lattice node.
    angles = jax.random.uniform(key, (cells, cells), maxval=2.0 * jnp.pi)
    gx, gy = jnp.cos(angles), jnp.sin(angles)  # (cells, cells)

    lin = jnp.linspace(0.0, float(steps), res)
    xx, yy = jnp.meshgrid(lin, lin, indexing="xy")
    xi = jnp.floor(xx).astype(jnp.int32)
    yi = jnp.floor(yy).astype(jnp.int32)
    xf, yf = xx - xi, yy - yi

    def dot(ix_off, iy_off):
        ix = jnp.clip(xi + ix_off, 0, cells - 1)
        iy = jnp.clip(yi + iy_off, 0, cells - 1)
        gxv = gx[iy, ix]
        gyv = gy[iy, ix]
        dxv = xx - (xi + ix_off).astype(jnp.float32)
        dyv = yy - (yi + iy_off).astype(jnp.float32)
        return gxv * dxv + gyv * dyv

    n00 = dot(0, 0)
    n10 = dot(1, 0)
    n01 = dot(0, 1)
    n11 = dot(1, 1)
    u = _fade(xf)
    v = _fade(yf)
    return _lerp(_lerp(n00, n10, u), _lerp(n01, n11, u), v)


def perlin_background(
    key: jax.Array, res: int, *, steps: int = 6, contrast: float = 0.3, baseline: float = 0.5
) -> Array:
    """A grayscale background field in [0,1] with random Perlin texture."""
    k_noise, k_c, k_b = jax.random.split(key, 3)
    noise = perlin_noise(k_noise, res, steps)
    c = contrast * (0.5 + jax.random.uniform(k_c, ()))
    b = baseline + 0.05 * (2.0 * jax.random.uniform(k_b, ()) - 1.0)
    return jnp.clip(b + c * noise, 0.0, 1.0)
