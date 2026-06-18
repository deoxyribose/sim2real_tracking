"""Many similar circular cells doing a random walk.

Inspired by the cells/dividing_cells_video.py generative model, but trimmed to:
- circular soft-mask cells (no FFT shape),
- per-cell random walk in tanh-space position,
- per-cell brightness with momentum drift,
- no division.

Cell ids are stable across frames (slot = identity).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sim2real.sim.configs import ManyCellsConfig
from sim2real.sim.primitives import gaussian_blob, make_grid
from sim2real.sim.render_common import (
    add_observation_noise,
    composite_video_frame,
    pack_zwhere,
    perlin_grayscale_bg,
)
from sim2real.types import SimSample

Array = jnp.ndarray


def _walk(key: jax.Array, init_xy: Array, T: int, step: float) -> Array:
    """Random walk in [-1,1]² clipped to (-0.95, 0.95). Returns (T, N, 2)."""
    deltas = jax.random.normal(key, (T, *init_xy.shape)) * step
    deltas = deltas.at[0].set(jnp.zeros_like(deltas[0]))
    xy = init_xy[None] + jnp.cumsum(deltas, axis=0)
    return jnp.clip(xy, -0.95, 0.95)


def sample(key: jax.Array, cfg: ManyCellsConfig) -> SimSample:
    cm = cfg.common
    n_max = cm.n_max
    n_active = cfg.n_objects

    k_pos, k_walk, k_r, k_color, k_color_walk, k_bg, k_noise, k_style = jax.random.split(key, 8)

    init_x = jax.random.uniform(k_pos, (n_max,), minval=-0.85, maxval=0.85)
    init_y = jax.random.uniform(
        jax.random.fold_in(k_pos, 1), (n_max,), minval=-0.85, maxval=0.85
    )
    init_xy = jnp.stack([init_x, init_y], axis=-1)                                          # (N, 2)
    radii = jax.random.uniform(k_r, (n_max,), minval=cfg.radius_min, maxval=cfg.radius_max) # (N,)
    base_color = jnp.clip(
        cfg.cell_color_mean + cfg.cell_color_std * jax.random.normal(k_color, (n_max,)),
        0.1, 1.0,
    )

    xy_traj = _walk(k_walk, init_xy, cm.T, cfg.walk_step)                                   # (T, N, 2)

    # Color drift across frames (small momentum walk).
    color_deltas = 0.02 * jax.random.normal(k_color_walk, (cm.T, n_max))
    color_deltas = color_deltas.at[0].set(jnp.zeros_like(color_deltas[0]))
    color_traj = jnp.clip(base_color[None] + jnp.cumsum(color_deltas, axis=0), 0.1, 1.0)    # (T, N)

    # Build masks per frame per slot via gaussian blob.
    xx, yy = make_grid(cm.res)

    def render_frame(t_idx, k_n):
        xy_t = xy_traj[t_idx]
        col_t = color_traj[t_idx]

        def one_slot(cx, cy, r):
            return gaussian_blob(xx, yy, cx, cy, r)

        masks = jax.vmap(one_slot)(xy_t[:, 0], xy_t[:, 1], radii)                          # (N, H, W)
        appear = masks[..., None] * col_t[:, None, None, None]                              # (N, H, W, 1)
        z_pres_t = jnp.concatenate([jnp.ones(n_active), jnp.zeros(n_max - n_active)])
        frame = composite_video_frame(masks, appear, bg, z_pres_t)
        return frame, masks

    bg = perlin_grayscale_bg(k_bg, cm.res, channels=1)
    keys_n = jax.random.split(k_noise, cm.T)
    t_idx = jnp.arange(cm.T)
    frames, masks = jax.vmap(render_frame)(t_idx, keys_n)                                   # (T, ...) , (T, N, H, W)
    video = jax.vmap(add_observation_noise, in_axes=(0, 0, None))(keys_n, frames, cm.obs_noise)

    # 5-dim z_where: round cells → sx = sy = radius, theta = 0.
    radii_TN = jnp.broadcast_to(radii[None, :], (cm.T, n_max))
    theta_TN = jnp.zeros_like(radii_TN)
    z_where = pack_zwhere(radii_TN, radii_TN, theta_TN, xy_traj[..., 0], xy_traj[..., 1])
    z_pres = jnp.broadcast_to(
        jnp.concatenate([jnp.ones(n_active), jnp.zeros(n_max - n_active)])[None, :],
        (cm.T, n_max),
    )

    z_style = jax.random.normal(k_style, (cm.style_dim,))

    return SimSample(
        video=video,
        z_where=z_where,
        z_pres=z_pres,
        z_style=z_style,
        masks=masks,
        z_what=None,
        meta={"sim_kind": "many_cells", "n_active": int(n_active)},
    )


def sample_batch(key: jax.Array, cfg: ManyCellsConfig, batch_size: int) -> SimSample:
    keys = jax.random.split(key, batch_size)
    return jax.vmap(sample, in_axes=(0, None))(keys, cfg)
