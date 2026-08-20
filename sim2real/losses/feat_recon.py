"""DINOSAUR-lite: reconstruct FROZEN target-encoder features from slot latents.

The insight (Seitzer et al 2023, arXiv 2209.14860): reconstructing per-pixel features has
NO "output flat background" shortcut, because the target features have per-pixel structure
(cell regions look different from background regions in feature space). Slots that opt out
(z_pres→0, z_where off-canvas) cannot escape this — a uniform feature map still mismatches
the structured target.

For our sim2real adaptation, this replaces pixel-space `recon_mse(composite, video)` with
`feat_mse(broadcast_decode(slots), frozen_target_encoder(video))`. Zero new heavy modules —
the "broadcast decoder" is a Gaussian mixture over slot positions with a single Dense-from-
z_what per-slot feature vector.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn

Array = jnp.ndarray


def _normalized_grid(h: int, w: int) -> Array:
    ys = jnp.linspace(-1.0, 1.0, h)
    xs = jnp.linspace(-1.0, 1.0, w)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    return jnp.stack([xx, yy], axis=-1).reshape(h * w, 2)                                  # (L, 2)


def broadcast_decode_to_feat(
    z_what: Array,      # (N, Zw)
    z_where: Array,     # (N, 5): (sx_raw, sy_raw, theta_raw, tx_raw, ty_raw)
    z_pres: Array,      # (N,)
    slot_to_feat_kernel: Array,   # (Zw, D_feat) — Dense weights, no bias (learned)
    H: int, W: int,
    scale_boost: float = 1.0,
    z_pres_temp: float = 1.0,
) -> Array:
    """Broadcast decode: sum_i softmax_over_slots(alpha_i(l)) * f(z_what_i)  →  (H, W, D_feat).

    alpha_i(l) = -0.5 * (pixel_pos(l) - slot_pos_i)^T diag(1/sigma_i)^2 (pixel_pos(l) - slot_pos_i)
                 + log(z_pres_i)      (softmax-normalized over slots)

    Uses supplied per-slot `slot_to_feat_kernel` (Zw × D_feat) — passed as an argument so this
    function is pure JAX; the trainable weights live in a Flax module (see FeatDecoder below).
    """
    N = z_what.shape[0]
    D_feat = slot_to_feat_kernel.shape[-1]
    L = H * W

    # Per-slot feature vector (Dense from z_what, no bias).
    slot_feats = z_what @ slot_to_feat_kernel                                              # (N, D_feat)

    # Per-slot pose (decoded from z_where).
    slot_pos = jnp.stack([jnp.tanh(z_where[:, 3]), jnp.tanh(z_where[:, 4])], axis=-1)     # (N, 2)
    sx = jnp.clip(jax.nn.sigmoid(z_where[:, 0]) * scale_boost, 0.02, 1.0)                  # (N,)
    sy = jnp.clip(jax.nn.sigmoid(z_where[:, 1]) * scale_boost, 0.02, 1.0)

    # Per-slot Gaussian alpha at each pixel.
    pixel_pos = _normalized_grid(H, W)                                                     # (L, 2)
    d = pixel_pos[None] - slot_pos[:, None]                                                # (N, L, 2)
    d2 = (d[..., 0] ** 2) / (sx[:, None] ** 2 + 1e-4) + \
         (d[..., 1] ** 2) / (sy[:, None] ** 2 + 1e-4)                                     # (N, L)
    alpha_logit = -0.5 * d2 + jnp.log(z_pres + 1e-6)[:, None] / z_pres_temp                # (N, L)
    alpha = jax.nn.softmax(alpha_logit, axis=0)                                            # (N, L) — softmax over slots

    # Composite.
    pred_feats_flat = alpha.T @ slot_feats                                                 # (L, D_feat)
    return pred_feats_flat.reshape(H, W, D_feat)


class FeatDecoder(nn.Module):
    """Trainable Flax wrapper: owns the Dense(Zw -> D_feat) kernel used by the decoder.

    Instantiate once (in adapt), then call `feat_decoder.apply(params, z_what, z_where, z_pres, H, W)`
    to compute pred_feats. Kept minimal to avoid architectural surface — a single Dense.
    """

    z_what_dim: int
    feat_dim: int

    @nn.compact
    def __call__(self, z_what, z_where, z_pres, H: int, W: int) -> Array:
        kernel = self.param(
            "slot_to_feat", nn.initializers.normal(stddev=0.02),
            (self.z_what_dim, self.feat_dim),
        )
        return broadcast_decode_to_feat(z_what, z_where, z_pres, kernel, H, W)


def feat_recon_mse(pred_feats: Array, target_feats: Array) -> Array:
    """MSE loss between predicted and (frozen) target features."""
    return jnp.mean((pred_feats - jax.lax.stop_gradient(target_feats)) ** 2)
