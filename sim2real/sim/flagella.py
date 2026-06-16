"""Flagella-like simulator: a few elongated splines with internal sinusoidal beating, mostly static
positions. Inherits the algae spirit but uses a closed-form beating model (no eigenvalue solver) so
it is trivially `jit` and `vmap` friendly.

Per object parameters:
  base position (tx, ty), base angle θ,
  length L, beat amplitude A, temporal beat frequency f, beat phase φ,
  wavenumber k along arc length s.

Shape(s, t) = base + R(θ) @ [s, A · sin(2π f t + k s + φ)]   with s ∈ [0, L].
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sim2real.sim.configs import FlagellaConfig
from sim2real.sim.render_common import (
    add_observation_noise,
    composite_video_frame,
    perlin_grayscale_bg,
)
from sim2real.sim.splines import render_polyline_segments_sdf
from sim2real.types import SimSample

Array = jnp.ndarray


def _sample_params(key: jax.Array, cfg: FlagellaConfig):
    """Sample per-object static parameters for one video. Returns dict with arrays of shape (n_max,)."""
    n_max = cfg.common.n_max
    keys = jax.random.split(key, 8)
    tx = jax.random.uniform(keys[0], (n_max,), minval=-0.6, maxval=0.6)
    ty = jax.random.uniform(keys[1], (n_max,), minval=-0.6, maxval=0.6)
    theta = jax.random.uniform(keys[2], (n_max,), minval=-jnp.pi, maxval=jnp.pi)
    L = jax.random.uniform(keys[3], (n_max,), minval=cfg.length_min, maxval=cfg.length_max)
    A = jax.random.uniform(keys[4], (n_max,), minval=cfg.amp_min, maxval=cfg.amp_max)
    f = jax.random.uniform(keys[5], (n_max,), minval=cfg.freq_min, maxval=cfg.freq_max)
    phi = jax.random.uniform(keys[6], (n_max,), minval=0.0, maxval=2.0 * jnp.pi)
    k_wave = jax.random.uniform(keys[7], (n_max,), minval=4.0, maxval=10.0)
    return {"tx": tx, "ty": ty, "theta": theta, "L": L, "A": A, "f": f, "phi": phi, "k": k_wave}


def _build_object_curves(params, t_norm: Array, n_points: int) -> Array:
    """Compute (n_max, n_points, 2) curve coordinates at normalized time `t_norm` ∈ [0, 1]."""
    s = jnp.linspace(0.0, 1.0, n_points)  # shared arc-length parameter
    sL = s[None, :] * params["L"][:, None]                                            # (N, P)
    lateral = params["A"][:, None] * jnp.sin(
        2.0 * jnp.pi * params["f"][:, None] * t_norm
        + params["k"][:, None] * sL
        + params["phi"][:, None]
    )                                                                                  # (N, P)
    # Rotate (sL, lateral) by theta and offset by (tx, ty).
    cos = jnp.cos(params["theta"])[:, None]
    sin = jnp.sin(params["theta"])[:, None]
    x = params["tx"][:, None] + cos * sL - sin * lateral
    y = params["ty"][:, None] + sin * sL + cos * lateral
    return jnp.stack([x, y], axis=-1)  # (N, P, 2)


def _z_where_from_curves(curves: Array) -> Array:
    """Derive z_where = (s_raw, tx_raw, ty_raw) from curve extents.

    - tx_raw, ty_raw = atanh of centroid (clipped to ±0.99).
    - s_raw = logit of half-extent (clipped to (0, 0.95)).
    """
    centroid = jnp.mean(curves, axis=-2)                          # (N, 2)
    half = 0.5 * (jnp.max(curves, axis=-2) - jnp.min(curves, axis=-2))  # (N, 2)
    extent = jnp.max(half, axis=-1)                               # (N,)
    txy = jnp.clip(centroid, -0.99, 0.99)
    tx_raw, ty_raw = jnp.arctanh(txy[:, 0]), jnp.arctanh(txy[:, 1])
    s = jnp.clip(extent, 1e-3, 0.95)
    s_raw = jax.scipy.special.logit(s)
    return jnp.stack([s_raw, tx_raw, ty_raw], axis=-1)            # (N, 3)


def sample(key: jax.Array, cfg: FlagellaConfig) -> SimSample:
    """Sample one flagella video. All shapes leading with T then N (no batch dim)."""
    cm = cfg.common
    k_static, k_bg, k_noise, k_style = jax.random.split(key, 4)

    static = _sample_params(k_static, cfg)
    n_max = cm.n_max
    n_active = cfg.n_objects
    # z_pres: first n_active are alive, rest dead. Order does not encode identity beyond slot index.
    z_pres_t = jnp.concatenate([jnp.ones(n_active), jnp.zeros(n_max - n_active)])
    z_pres = jnp.broadcast_to(z_pres_t, (cm.T, n_max))                                   # (T, N)

    # Per-frame curves + masks
    def per_frame(t_idx):
        t_norm = t_idx / (cm.T - 1)
        curves = _build_object_curves(static, t_norm, cfg.n_points)                       # (N, P, 2)
        # Render mask per object (vmap over slot)
        masks = jax.vmap(lambda c: render_polyline_segments_sdf(c, cm.res, cfg.line_sigma_px))(curves)
        z_where_t = _z_where_from_curves(curves)
        return curves, masks, z_where_t

    t_idx = jnp.arange(cm.T)
    curves, masks, z_where = jax.vmap(per_frame)(t_idx)                                   # (T, N, ...)

    # Render frames.
    # Background sampled once per video (no per-frame shake here — flagella are mostly static).
    bg = perlin_grayscale_bg(k_bg, cm.res, channels=1)                                    # (H, W, 1)
    color = jnp.array([1.0])                                                              # bright flagella on dark bg

    def render_frame(masks_t, zp_t, k_n):
        appear = masks_t[..., None] * color[None, None, None, :]                          # (N, H, W, 1)
        frame = composite_video_frame(masks_t, appear, bg, zp_t)
        return add_observation_noise(k_n, frame, cm.obs_noise)

    keys_n = jax.random.split(k_noise, cm.T)
    video = jax.vmap(render_frame)(masks, z_pres, keys_n)                                 # (T, H, W, 1)

    z_style = jax.random.normal(k_style, (cm.style_dim,))                                 # placeholder GT

    return SimSample(
        video=video,
        z_where=z_where,
        z_pres=z_pres,
        z_style=z_style,
        masks=masks,
        z_what=None,
        meta={"sim_kind": "flagella", "n_active": int(n_active)},
    )


def sample_batch(key: jax.Array, cfg: FlagellaConfig, batch_size: int) -> SimSample:
    """vmap helper — returns a SimSample with a leading batch dim."""
    keys = jax.random.split(key, batch_size)
    return jax.vmap(sample, in_axes=(0, None))(keys, cfg)
