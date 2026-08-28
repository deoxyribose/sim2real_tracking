"""Structural cell-anchored flagellum head (approach #3).

Reparameterize the flagellum attachment as:
    (cell_slot_index probs, angle_around_cell, radius_offset)

An attention softmax over cell slots decides which cell the flagellum attaches to.
The attachment y/x are DERIVED from the chosen cell's center + radius + angle, not
predicted directly. This makes the attach-on-cell constraint STRUCTURALLY
guaranteed rather than statistically learned.

For control points, we still predict K arc-length points as Gaussian offsets from
the derived attachment, so the flagellum can curve freely once attached.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..data import CANONICAL_H, CANONICAL_W, FLAGELLUM_K


class AnchoredFlagellumHead(nn.Module):
    """Predict flagellum params with structural cell attachment.

    Given (slots, cell_head_out), each flagellum slot predicts:
      - anchor_logits: (B, S, S_cell)  attention over cell slots
      - anchor_angle:  (B, S) in [0, 2π)  where on the chosen cell's boundary
      - anchor_radius_scale: (B, S) in [0.7, 1.3]  slight radial offset from cell.radius
      - control_offsets_mean, log_sigma: (B, S, K, 2)  offsets from attachment
      - width, amp, polarity — same as FlagellumHead

    At sampling time:
      cell_slot_j ~ Categorical(softmax(anchor_logits))
      attachment = cell_center[j] + (anchor_radius_scale * cell_radius[j]) * (sin θ, cos θ)
      control_points[k] = attachment + cumulative(offset_deltas)
    """
    d_ff: int = 256

    @nn.compact
    def __call__(self, slots: jnp.ndarray, cell_out: dict) -> dict:
        B, S, D = slots.shape
        S_cell = cell_out["cell_center_mean"].shape[1]  # same as S here (all slots can be cells)
        x = nn.Dense(self.d_ff, name="fc1")(slots)
        x = jax.nn.gelu(x)
        x = nn.Dense(self.d_ff, name="fc2")(x)
        x = jax.nn.gelu(x)

        # Anchor: attend to cell candidate slots
        anchor_logits = nn.Dense(S_cell, name="anchor_logits")(x)   # (B, S, S_cell)
        # Angle around cell (parametrize as (sin, cos) then atan for numerical stability)
        angle_sc = nn.Dense(2, name="angle_sc")(x)                   # (B, S, 2) — unnormalized (sin, cos)
        norm = jnp.linalg.norm(angle_sc, axis=-1, keepdims=True) + 1e-6
        angle_sc = angle_sc / norm                                    # unit vector on circle
        angle_rad = jnp.arctan2(angle_sc[..., 0], angle_sc[..., 1])   # (B, S) in (-π, π]

        # Radial scale (mostly 1.0, tiny drift)
        rad_scale = 0.85 + 0.30 * jax.nn.sigmoid(nn.Dense(1, name="rad_scale")(x)[..., 0])  # in [0.85, 1.15]

        # Derive attachment (soft — weight by anchor_logits softmax)
        anchor_probs = jax.nn.softmax(anchor_logits, axis=-1)  # (B, S, S_cell)
        # expected cell center + radius weighted by anchor probs
        cell_center = cell_out["cell_center_mean"]              # (B, S_cell, 2)
        cell_radius = cell_out["cell_radius_mean"]              # (B, S_cell)
        exp_center = jnp.einsum("bsc,bcd->bsd", anchor_probs, cell_center)     # (B, S, 2)
        exp_radius = jnp.einsum("bsc,bc->bs", anchor_probs, cell_radius)       # (B, S)
        effective_r = exp_radius * rad_scale                    # (B, S)
        attach_y = exp_center[..., 0] + effective_r * jnp.sin(angle_rad)
        attach_x = exp_center[..., 1] + effective_r * jnp.cos(angle_rad)
        attachment_mean = jnp.stack([attach_y, attach_x], axis=-1)              # (B, S, 2)

        # Attachment uncertainty (Gaussian around derived point)
        attach_log_sigma = -1.0 + 2.5 * jax.nn.tanh(nn.Dense(2, name="attach_log_sigma")(x))  # in [-3.5, 1.5]

        # Control-point offsets from attachment (cumulative K, K arc-length steps)
        cp_off_mean = nn.Dense(FLAGELLUM_K * 2, name="cp_off_mean")(x).reshape(B, S, FLAGELLUM_K, 2) * 10.0
        cp_off_log_sigma = 0.0 + 3.0 * jax.nn.tanh(
            nn.Dense(FLAGELLUM_K * 2, name="cp_off_log_sigma")(x).reshape(B, S, FLAGELLUM_K, 2)
        )   # in [-3, 3] → σ in [0.05, 20] per step (before k_shift)

        # Auxiliary scalars — same as FlagellumHead
        aux = nn.Dense(5, name="aux")(x)
        width_mean = 2.5 + 6.0 * jax.nn.sigmoid(aux[..., 0])
        width_log_sigma = 0.0 + 3.0 * jax.nn.tanh(aux[..., 1])
        amp_mean = 2.0 + 15.0 * jax.nn.sigmoid(aux[..., 2])
        amp_log_sigma = 1.0 + 3.0 * jax.nn.tanh(aux[..., 3])
        polarity_logit = aux[..., 4]

        # Convert offset deltas → cumulative absolute control points (for parity with FlagellumHead API)
        cp_cum = jnp.cumsum(cp_off_mean, axis=2) + attachment_mean[:, :, None]   # (B, S, K, 2)
        pts_mean = jnp.concatenate([attachment_mean[:, :, None], cp_cum], axis=2)  # (B, S, K+1, 2)

        # Combined sigma per point: attachment has its own, control points share their per-step sigma
        pts_log_sigma_att = attach_log_sigma[:, :, None]                          # (B, S, 1, 2)
        pts_log_sigma_cp = cp_off_log_sigma                                        # (B, S, K, 2)
        # For cumulative offsets, sigma grows sqrt(k) — approximate with 0.5*log(k+1) shift
        k_shift = 0.5 * jnp.log(jnp.arange(1, FLAGELLUM_K + 1) + 1)[None, None, :, None]  # (1, 1, K, 1)
        pts_log_sigma_cp = pts_log_sigma_cp + k_shift
        pts_log_sigma = jnp.concatenate([pts_log_sigma_att, pts_log_sigma_cp], axis=2)

        return dict(
            pts_mean=pts_mean, pts_log_sigma=pts_log_sigma,
            width_mean=width_mean, width_log_sigma=width_log_sigma,
            amp_mean=amp_mean, amp_log_sigma=amp_log_sigma,
            polarity_logit=polarity_logit,
            # New diagnostic outputs (not consumed by loss but useful for viz)
            anchor_probs=anchor_probs,
            anchor_angle=angle_rad,
        )
