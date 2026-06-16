"""Same-shape, varied-pose worms.

A shared learned-ish shape template (a curve through K control points, sampled once per video)
is *replicated* across all worms with per-worm (rotation, scale, translation) pose, plus a small
amount of bending to keep them non-identical.

This tests z_what cross-object similarity prior — all worms should encode to similar z_what,
differing mainly in pose (which lives in z_where).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sim2real.sim.configs import WormsConfig
from sim2real.sim.render_common import (
    add_observation_noise,
    composite_video_frame,
    perlin_grayscale_bg,
)
from sim2real.sim.splines import catmull_rom_eval, render_polyline_segments_sdf
from sim2real.types import SimSample


def _sample_template(key, cfg: WormsConfig):
    """Sample one shared shape template — a curve in local coordinates [-1, 1] × [-amp, amp]."""
    k1, k2 = jax.random.split(key, 2)
    # Control points: x evenly along [-1, 1]; y small zigzag.
    xs = jnp.linspace(-1.0, 1.0, cfg.n_ctrl)
    ys = 0.3 * jax.random.normal(k1, (cfg.n_ctrl,))
    ctrl = jnp.stack([xs, ys], axis=-1)
    curve = catmull_rom_eval(ctrl, cfg.n_points)                # (n_points, 2)
    # Scale to nominal length cfg.length / 2 (half-length).
    half = cfg.length / 2.0
    curve = curve * jnp.array([half, half * 0.4])
    del k2
    return curve                                                # (n_points, 2)


def _per_worm_pose(key, cfg: WormsConfig, n_active: int):
    """Sample per-worm initial pose: (tx, ty), rotation θ, small bend amplitude."""
    n_max = cfg.common.n_max
    keys = jax.random.split(key, 4)
    tx = jax.random.uniform(keys[0], (n_max,), minval=-0.7, maxval=0.7)
    ty = jax.random.uniform(keys[1], (n_max,), minval=-0.7, maxval=0.7)
    theta = jax.random.uniform(keys[2], (n_max,), minval=-jnp.pi, maxval=jnp.pi)
    bend = 0.04 * jax.random.normal(keys[3], (n_max,))
    # Zero out the dead-slot pose to keep z_where in-range (does not affect z_pres=0 slots' loss).
    mask = jnp.arange(n_max) < n_active
    return tx * mask, ty * mask, theta * mask, bend * mask


def sample(key: jax.Array, cfg: WormsConfig) -> SimSample:
    cm = cfg.common
    n_max = cm.n_max
    n_active = cfg.n_objects

    k_tpl, k_pose, k_walk, k_rot, k_bg, k_noise, k_style = jax.random.split(key, 7)
    template = _sample_template(k_tpl, cfg)                                                 # (P, 2)

    tx0, ty0, th0, bend = _per_worm_pose(k_pose, cfg, n_active)

    # Walks
    wxy = jax.random.normal(k_walk, (cm.T, n_max, 2)) * cfg.walk_step
    wxy = wxy.at[0].set(jnp.zeros_like(wxy[0]))
    xy_traj = jnp.stack([tx0, ty0], axis=-1)[None] + jnp.cumsum(wxy, axis=0)                 # (T, N, 2)
    xy_traj = jnp.clip(xy_traj, -0.9, 0.9)

    rot_w = jax.random.normal(k_rot, (cm.T, n_max)) * cfg.rot_step
    rot_w = rot_w.at[0].set(jnp.zeros_like(rot_w[0]))
    theta_traj = th0[None] + jnp.cumsum(rot_w, axis=0)                                       # (T, N)

    # Per-worm curve: template + bend × sin(πs) lateral, then rotate + translate.
    s = jnp.linspace(-1.0, 1.0, cfg.n_points)
    lateral_bias = jnp.sin(jnp.pi * (s + 1.0) / 2.0)[None, :]                                # (1, P)

    def build_curve(theta, bend_amp, xy_t):
        # Bent template in local coords: (P, 2).
        local_y = template[:, 1] + bend_amp * lateral_bias[0]
        local = jnp.stack([template[:, 0], local_y], axis=-1)
        c, s_ = jnp.cos(theta), jnp.sin(theta)
        rot = jnp.array([[c, -s_], [s_, c]])
        rotated = local @ rot.T
        return rotated + xy_t                                                                # (P, 2)

    def per_frame(t_idx):
        def per_slot(theta, bend_amp, xy_t):
            return build_curve(theta, bend_amp, xy_t)

        curves = jax.vmap(per_slot)(theta_traj[t_idx], bend, xy_traj[t_idx])                 # (N, P, 2)
        masks = jax.vmap(lambda c: render_polyline_segments_sdf(c, cm.res, cfg.width_px))(curves)
        return curves, masks

    curves, masks = jax.vmap(per_frame)(jnp.arange(cm.T))                                    # (T, N, ...)

    bg = perlin_grayscale_bg(k_bg, cm.res, channels=1)
    color = jnp.array([0.9])

    def render_frame(masks_t, k_n):
        appear = masks_t[..., None] * color[None, None, None, :]
        z_pres_t = jnp.concatenate([jnp.ones(n_active), jnp.zeros(n_max - n_active)])
        frame = composite_video_frame(masks_t, appear, bg, z_pres_t)
        return add_observation_noise(k_n, frame, cm.obs_noise)

    keys_n = jax.random.split(k_noise, cm.T)
    video = jax.vmap(render_frame)(masks, keys_n)                                            # (T, H, W, 1)

    # z_where: derived from curve extents per frame.
    def zwhere_from_curves(c):
        centroid = jnp.mean(c, axis=0)
        half = 0.5 * (jnp.max(c, axis=0) - jnp.min(c, axis=0))
        extent = jnp.max(half)
        txy = jnp.clip(centroid, -0.99, 0.99)
        s_clipped = jnp.clip(extent, 1e-3, 0.95)
        return jnp.array([jax.scipy.special.logit(s_clipped), jnp.arctanh(txy[0]), jnp.arctanh(txy[1])])

    z_where = jax.vmap(jax.vmap(zwhere_from_curves))(curves)                                 # (T, N, 3)
    z_pres = jnp.broadcast_to(
        jnp.concatenate([jnp.ones(n_active), jnp.zeros(n_max - n_active)])[None, :],
        (cm.T, n_max),
    )
    z_style = jax.random.normal(k_style, (cm.style_dim,))

    # The shared template is the per-video appearance prior — expose it as the GT z_what,
    # broadcast across slots, so the cross-object KL has a ground-truth to compare against.
    template_flat = template.reshape(-1)                                                     # (P*2,)
    z_what = jnp.broadcast_to(template_flat[None, :], (n_max, template_flat.shape[0]))

    return SimSample(
        video=video,
        z_where=z_where,
        z_pres=z_pres,
        z_style=z_style,
        masks=masks,
        z_what=z_what,
        meta={"sim_kind": "worms", "n_active": int(n_active)},
    )


def sample_batch(key: jax.Array, cfg: WormsConfig, batch_size: int) -> SimSample:
    keys = jax.random.split(key, batch_size)
    return jax.vmap(sample, in_axes=(0, None))(keys, cfg)
