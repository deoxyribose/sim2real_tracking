"""JIT'd, batched AR sampler shared by viz/eval scripts.

`sample_pool_one_clip(...)` runs on ONE clip and:
  1. Builds a batch of TTA-augmented views (angles × flips × noise draws)
  2. Runs a single backbone forward on that batch
  3. Samples top-K attachments per view (host side, small tensor)
  4. Runs rollouts for all (view, attach) pairs in ONE vmapped scan
  5. Inverse-transforms rollouts back to model canvas coords

Falls back on cv2 for the rotation (host side, negligible cost) so the
pure-JAX section is one large graph.
"""
from __future__ import annotations

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from sim2real.model.unet_ar import (
    AttachmentHead, KnotGenerator, UNetARBackbone, UNetARConfig,
    rotated_patch_batched, sample_batched_noise, unpack_attachment,
)


def _rotate_np(clip: np.ndarray, angle_deg: float, flipped: bool) -> np.ndarray:
    """Rotate + flip a (T, H, W) or (H, W) numpy clip. Host-side."""
    if flipped:
        clip = clip[..., ::-1].copy()
    if angle_deg != 0:
        H, W = clip.shape[-2], clip.shape[-1]
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle_deg, 1.0)
        if clip.ndim == 3:
            clip = np.stack([cv2.warpAffine(clip[t], M, (W, H))
                              for t in range(clip.shape[0])], axis=0)
        else:
            clip = cv2.warpAffine(clip, M, (W, H))
    return clip


def _inv_rotate_pts(pts: np.ndarray, H: int, W: int,
                      angle_deg: float, flipped: bool) -> np.ndarray:
    """Un-augment (K, 2) yx points. Host side, negligible."""
    x = pts.copy()
    if flipped:
        x[:, 1] = (W - 1) - x[:, 1]
    if angle_deg != 0:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), -angle_deg, 1.0)
        pts3 = np.stack([x[:, 1], x[:, 0], np.ones(x.shape[0])], axis=-1)
        xy = pts3 @ M.T
        x = np.stack([xy[:, 1], xy[:, 0]], axis=-1)
    return x


def make_sampler(cfg: UNetARConfig, backbone, attach_head, knot_gen,
                   temperature: float = 1.0):
    """Return a JIT'd (params, videos, noises, smeds) → (full_res, grid)
    plus a JIT'd rollout_scan function.

    Videos/noises/smeds should already be batched (B, ...) where B = views × draws.
    temperature < 1 → peakier categorical (mode-centred). temp=0 → argmax.
    """
    @jax.jit
    def backbone_batched(params, videos, noises, smeds):
        full_res, grid = backbone.apply(params["backbone"], videos, noises,
                                          smeds, train=False)
        attach = attach_head.apply(params["attach"], grid)
        return full_res, attach   # attach: (B, gh, gw, n_sug, 3)

    @jax.jit
    def rollout_scan(feature_maps, params_knot, attaches, keys):
        """feature_maps: (B, H, W, C); attaches: (B, M, 2); keys: (B, M)."""
        def per_view(feature_map, atts, ks):
            def per_attach(att, k):
                def step(carry, _):
                    pos, tangent, kk = carry
                    patch = rotated_patch_batched(feature_map, pos[None],
                                                    tangent[None], cfg.patch_size)
                    a_lg, s_lg = knot_gen.apply(params_knot, patch)
                    ka = jax.random.fold_in(k, kk)
                    ks_ = jax.random.fold_in(k, kk + 100000)
                    if temperature == 0.0:
                        a_bin = jnp.argmax(a_lg[0], axis=-1)
                        s_bin = jnp.argmax(s_lg[0], axis=-1)
                    else:
                        a_bin = jax.random.categorical(ka, a_lg[0] / temperature)
                        s_bin = jax.random.categorical(ks_, s_lg[0] / temperature)
                    d_ang = cfg.angle_bin_centers[a_bin]
                    d_step = cfg.step_bin_centers[s_bin]
                    new_tan = tangent + d_ang
                    new_pos = jnp.stack([pos[0] + d_step * jnp.sin(new_tan),
                                          pos[1] + d_step * jnp.cos(new_tan)])
                    return (new_pos, new_tan, kk + 1), new_pos
                (_, _, _), positions = jax.lax.scan(
                    step, (att, jnp.array(0.0), jnp.int32(0)), jnp.arange(cfg.n_knots))
                return jnp.concatenate([att[None], positions], axis=0)
            return jax.vmap(per_attach)(atts, ks)
        return jax.vmap(per_view)(feature_maps, attaches, keys)

    return backbone_batched, rollout_scan


def sample_pool_one_clip(params, backbone, attach_head, knot_gen, cfg,
                           clip: np.ndarray, smed: np.ndarray,
                           tta_angles: list[float], flips=(False, True),
                           n_draws: int = 2, n_attach: int = 8,
                           n_rollouts: int = 1,
                           score_thresh: float = 0.02,
                           key: jax.Array = None,
                           _sampler=None,
                           return_scores: bool = False):
    """Return list of rollouts (each (K+1, 2) in model canvas coords)."""
    if _sampler is None:
        _sampler = make_sampler(cfg, backbone, attach_head, knot_gen)
    backbone_batched, rollout_scan = _sampler
    if key is None:
        key = jax.random.key(0)

    # Build augmented batch
    views_clip, views_smed, angle_flip = [], [], []
    for angle in tta_angles:
        for flipped in flips:
            views_clip.append(_rotate_np(clip, angle, flipped))
            views_smed.append(_rotate_np(smed, angle, flipped))
            angle_flip.append((angle, flipped))
    V = len(views_clip)                            # number of TTA views
    B = V * n_draws
    videos = jnp.stack([jnp.asarray(v) for v in views_clip], axis=0)  # (V, T, H, W)
    smeds  = jnp.stack([jnp.asarray(v)[..., None] for v in views_smed], axis=0)  # (V, H, W, 1)
    # Repeat each view n_draws times
    videos = jnp.repeat(videos, n_draws, axis=0)   # (B, T, H, W)
    smeds  = jnp.repeat(smeds,  n_draws, axis=0)

    key, k_n = jax.random.split(key)
    noises = sample_batched_noise(k_n, B, cfg)     # (B, H, W, 1)

    full_res, attach = backbone_batched(params, videos, noises, smeds)
    full_res = np.asarray(full_res)                # (B, H, W, C)
    attach   = np.asarray(attach)                  # (B, gh, gw, n_sug, 3)

    # Pick top-K attachments per view (host, small tensors)
    ys = np.arange(cfg.grid_h) * cfg.grid_stride + cfg.grid_stride / 2
    xs = np.arange(cfg.grid_w) * cfg.grid_stride + cfg.grid_stride / 2
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    cell_centers = np.stack([yy, xx], -1)          # (gh, gw, 2)

    all_attaches, all_keep_mask, all_scores = [], [], []
    for b in range(B):
        raw = attach[b]                            # (gh, gw, n_sug, 3)
        dy = 32.0 * np.tanh(raw[..., 0])
        dx = 32.0 * np.tanh(raw[..., 1])
        sc = 1.0 / (1.0 + np.exp(-raw[..., 2]))
        pos = cell_centers[:, :, None, :] + np.stack([dy, dx], -1)
        flat_pos = pos.reshape(-1, 2)
        flat_sc  = sc.reshape(-1)
        idx = np.argsort(-flat_sc)[:n_attach]
        keep = flat_sc[idx] >= score_thresh
        atts = flat_pos[idx].astype(np.float32)    # (n_attach, 2)
        sc_top = flat_sc[idx].astype(np.float32)
        atts_kept = np.where(keep[:, None], atts, atts[0:1])
        all_attaches.append(atts_kept)
        all_keep_mask.append(keep)
        all_scores.append(sc_top)

    all_attaches  = np.stack(all_attaches, axis=0)   # (B, n_attach, 2)
    all_keep_mask = np.stack(all_keep_mask, axis=0)
    all_scores    = np.stack(all_scores, axis=0)     # (B, n_attach)

    out, scores_out = [], []
    for r in range(n_rollouts):
        key, kr = jax.random.split(key)
        keys_flat = jax.random.split(kr, B * n_attach).reshape(B, n_attach)
        rollouts = np.asarray(rollout_scan(
            jnp.asarray(full_res), params["knot"],
            jnp.asarray(all_attaches), keys_flat))
        for b in range(B):
            view_idx = b // n_draws
            angle, flipped = angle_flip[view_idx]
            for m in range(rollouts.shape[1]):
                if not all_keep_mask[b, m]:
                    continue
                inv = _inv_rotate_pts(rollouts[b, m], cfg.H, cfg.W, angle, flipped)
                out.append(inv)
                scores_out.append(float(all_scores[b, m]))
    if return_scores:
        return out, scores_out, key
    return out, key
