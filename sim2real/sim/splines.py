"""Spline helpers.

For the simulators we don't need the full `interpax` interpolator API at sample time —
we just need a small set of operations: build control points, evaluate along a parameterization,
and convert between bounded `knots` and unconstrained `params` (sigmoid / logit). This mirrors
the worms/utils.py convention so spline-fitting code from there can plug in cleanly later.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def params2knots(params: Array) -> Array:
    """Map unconstrained `params` ∈ ℝ to bounded `knots` ∈ (0, 1) via sigmoid."""
    return jax.nn.sigmoid(params)


def knots2params(knots: Array) -> Array:
    """Inverse of `params2knots` — logit."""
    return jax.scipy.special.logit(knots)


def catmull_rom_eval(control_pts: Array, n_samples: int) -> Array:
    """Evaluate a Catmull-Rom-like spline through `control_pts` at `n_samples` evenly-spaced
    parameter values.

    Args:
      control_pts: (K, D) — K control points in D dimensions (D=2 for x,y).
      n_samples: number of output samples.

    Returns:
      (n_samples, D) — the sampled curve.
    """
    K = control_pts.shape[0]
    # Evaluate by piecewise cubic Catmull-Rom segments. We use centripetal-style parameterization
    # but with uniform t spacing for simplicity (sufficient for smooth shape generation).
    t = jnp.linspace(0.0, K - 1.0, n_samples)
    i = jnp.clip(jnp.floor(t).astype(jnp.int32), 0, K - 2)
    u = (t - i)[:, None]

    # Indices with clamped boundaries
    i0 = jnp.clip(i - 1, 0, K - 1)
    i1 = i
    i2 = jnp.clip(i + 1, 0, K - 1)
    i3 = jnp.clip(i + 2, 0, K - 1)

    p0 = control_pts[i0]
    p1 = control_pts[i1]
    p2 = control_pts[i2]
    p3 = control_pts[i3]

    # Catmull-Rom basis (tension = 0.5)
    a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
    b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3
    c = -0.5 * p0 + 0.5 * p2
    d = p1
    return ((a * u + b) * u + c) * u + d


def arc_length_resample(curve: Array, n_samples: int) -> Array:
    """Resample a sampled curve so that points are equidistant in arc length."""
    seg = jnp.linalg.norm(curve[1:] - curve[:-1], axis=-1)
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)])
    total = cum[-1] + 1e-8
    s = jnp.linspace(0.0, total, n_samples)
    # For each target arc-length s_i, find the segment it falls in and interpolate.
    idx = jnp.clip(jnp.searchsorted(cum, s) - 1, 0, curve.shape[0] - 2)
    t = (s - cum[idx]) / (cum[idx + 1] - cum[idx] + 1e-8)
    p0 = curve[idx]
    p1 = curve[idx + 1]
    return p0 + t[:, None] * (p1 - p0)


def render_polyline_sdf(points: Array, res: int, sigma: float = 2.0) -> Array:
    """Render the polyline through `points` (N, 2) as an exponential-distance soft mask.

    Output: (res, res) values in [0, 1] — peaks at 1 on the polyline.
    Points should be in normalized [-1, 1] coordinates.
    """
    # Convert to pixel space
    px = (points + 1.0) * 0.5 * res
    lin = jnp.arange(res, dtype=jnp.float32)
    xx, yy = jnp.meshgrid(lin, lin, indexing="xy")
    # distance from each pixel to each polyline point
    dx = xx[..., None] - px[:, 0]
    dy = yy[..., None] - px[:, 1]
    min_d2 = jnp.min(dx * dx + dy * dy, axis=-1)
    return jnp.exp(-0.5 * min_d2 / (sigma ** 2 + 1e-8))


def render_polyline_segments_sdf(points: Array, res: int, sigma: float = 2.0) -> Array:
    """Same as `render_polyline_sdf` but distance to the polyline *segments*, not vertices.

    Produces smoother lines for sparse vertex sampling.
    """
    px = (points + 1.0) * 0.5 * res
    lin = jnp.arange(res, dtype=jnp.float32)
    xx, yy = jnp.meshgrid(lin, lin, indexing="xy")
    # For each segment (p_i, p_{i+1}) compute distance from pixels to segment
    a = px[:-1]
    b = px[1:]
    ab = b - a  # (S, 2)
    ab2 = jnp.sum(ab * ab, axis=-1) + 1e-8  # (S,)
    # pixel coords broadcast
    p_minus_a_x = xx[..., None] - a[:, 0]
    p_minus_a_y = yy[..., None] - a[:, 1]
    t = (p_minus_a_x * ab[:, 0] + p_minus_a_y * ab[:, 1]) / ab2  # (H,W,S)
    t = jnp.clip(t, 0.0, 1.0)
    proj_x = a[:, 0] + t * ab[:, 0]
    proj_y = a[:, 1] + t * ab[:, 1]
    d2 = (xx[..., None] - proj_x) ** 2 + (yy[..., None] - proj_y) ** 2
    min_d2 = jnp.min(d2, axis=-1)
    return jnp.exp(-0.5 * min_d2 / (sigma ** 2 + 1e-8))
