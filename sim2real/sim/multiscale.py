"""Multi-scale simulator: a few large objects (nuclei) + many small puncta in the same frame.

Tests heterogeneous scales — the z_where scale dimension has to cover a wide range, and the
discovery / propagation mechanisms must not be biased toward any one scale.

Slot ordering: large objects occupy slots [0 : n_large), small objects [n_large : n_large+n_small).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sim2real.sim.configs import MultiScaleConfig
from sim2real.sim.primitives import gaussian_blob, make_grid
from sim2real.sim.render_common import (
    add_observation_noise,
    composite_video_frame,
    pack_zwhere,
    perlin_grayscale_bg,
)
from sim2real.types import SimSample


def _walk(key, init_xy, T, step):
    deltas = jax.random.normal(key, (T, *init_xy.shape)) * step
    deltas = deltas.at[0].set(jnp.zeros_like(deltas[0]))
    return jnp.clip(init_xy[None] + jnp.cumsum(deltas, axis=0), -0.95, 0.95)


def sample(key: jax.Array, cfg: MultiScaleConfig) -> SimSample:
    cm = cfg.common
    n_max = cm.n_max
    n_large = cfg.n_large
    n_small = cfg.n_small
    n_active = n_large + n_small
    assert n_active <= n_max, "n_max too small for n_large + n_small"

    keys = jax.random.split(key, 9)
    init_x_l = jax.random.uniform(keys[0], (n_large,), minval=-0.5, maxval=0.5)
    init_y_l = jax.random.uniform(keys[1], (n_large,), minval=-0.5, maxval=0.5)
    init_x_s = jax.random.uniform(keys[2], (n_small,), minval=-0.85, maxval=0.85)
    init_y_s = jax.random.uniform(keys[3], (n_small,), minval=-0.85, maxval=0.85)
    r_l = jax.random.uniform(keys[4], (n_large,), minval=cfg.large_radius_min, maxval=cfg.large_radius_max)
    r_s = jax.random.uniform(keys[5], (n_small,), minval=cfg.small_radius_min, maxval=cfg.small_radius_max)

    # Pad to n_max
    pad_n = n_max - n_active
    init_xy_active = jnp.concatenate(
        [jnp.stack([init_x_l, init_y_l], axis=-1), jnp.stack([init_x_s, init_y_s], axis=-1)],
        axis=0,
    )                                                                                       # (n_active, 2)
    init_xy = jnp.concatenate([init_xy_active, jnp.zeros((pad_n, 2))], axis=0)               # (n_max, 2)
    radii_active = jnp.concatenate([r_l, r_s])
    radii = jnp.concatenate([radii_active, jnp.ones(pad_n) * cfg.small_radius_min])          # (n_max,)

    # Step sizes: large slow, small fast.
    step_active = jnp.concatenate(
        [jnp.full((n_large,), cfg.walk_step_large), jnp.full((n_small,), cfg.walk_step_small)]
    )
    step_arr = jnp.concatenate([step_active, jnp.zeros(pad_n)])                              # (n_max,)

    # Walk each slot with its own step. Use per-slot noise scaled by step_arr.
    raw = jax.random.normal(keys[6], (cm.T, n_max, 2)) * step_arr[None, :, None]
    raw = raw.at[0].set(jnp.zeros_like(raw[0]))
    xy_traj = jnp.clip(init_xy[None] + jnp.cumsum(raw, axis=0), -0.95, 0.95)                 # (T, N, 2)

    bg = perlin_grayscale_bg(keys[7], cm.res, channels=1)
    xx, yy = make_grid(cm.res)

    # Colors: large slightly darker, small bright.
    colors_active = jnp.concatenate(
        [jnp.full((n_large,), 0.6), jnp.full((n_small,), 0.95)]
    )
    colors = jnp.concatenate([colors_active, jnp.zeros(pad_n)])                              # (n_max,)

    def render_frame(t_idx, k_n):
        xy_t = xy_traj[t_idx]

        def one_slot(cx, cy, r):
            return gaussian_blob(xx, yy, cx, cy, r)

        masks = jax.vmap(one_slot)(xy_t[:, 0], xy_t[:, 1], radii)                          # (N, H, W)
        appear = masks[..., None] * colors[:, None, None, None]
        z_pres_t = jnp.concatenate([jnp.ones(n_active), jnp.zeros(n_max - n_active)])
        frame = composite_video_frame(masks, appear, bg, z_pres_t)
        return frame, masks

    keys_n = jax.random.split(keys[8], cm.T)
    t_idx = jnp.arange(cm.T)
    frames, masks = jax.vmap(render_frame)(t_idx, keys_n)
    video = jax.vmap(add_observation_noise, in_axes=(0, 0, None))(keys_n, frames, cm.obs_noise)

    radii_TN = jnp.broadcast_to(radii[None, :], (cm.T, n_max))
    theta_TN = jnp.zeros_like(radii_TN)
    z_where = pack_zwhere(
        radii_TN, radii_TN, theta_TN, xy_traj[..., 0], xy_traj[..., 1], scale_factor=2.5
    )
    z_pres = jnp.broadcast_to(
        jnp.concatenate([jnp.ones(n_active), jnp.zeros(n_max - n_active)])[None, :],
        (cm.T, n_max),
    )
    z_style = jax.random.normal(jax.random.fold_in(keys[7], 99), (cm.style_dim,))

    return SimSample(
        video=video,
        z_where=z_where,
        z_pres=z_pres,
        z_style=z_style,
        masks=masks,
        z_what=None,
        meta={"sim_kind": "multiscale", "n_large": int(n_large), "n_small": int(n_small)},
    )


def sample_batch(key: jax.Array, cfg: MultiScaleConfig, batch_size: int) -> SimSample:
    keys = jax.random.split(key, batch_size)
    return jax.vmap(sample, in_axes=(0, None))(keys, cfg)
