"""Typed distributional heads per slot.

Per slot:
  - class logits (N_CLASSES)
  - class-conditional parameter heads with mean + log_σ

For MVP only the flagellum head is fully implemented. Cell / pipette heads exist as
stubs (predict mean-only params, no NLL loss contribution) and can be filled in later
without changing the top-level model API.

Flagellum head predicts (attachment + K control points), each (y, x), each with a
predicted Gaussian mean and log_σ. Amplitude / width / polarity are auxiliary scalar
outputs (mean+log_σ each). All outputs are in CANONICAL PIXEL / σ units.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..data import CANONICAL_H, CANONICAL_W, FLAGELLUM_K, N_CLASSES


class ClassHead(nn.Module):
    """Per-slot class logits over {∅, pipette, cell, flagellum}."""
    d_ff: int = 128

    @nn.compact
    def __call__(self, slots: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.d_ff, name="fc1")(slots)
        x = jax.nn.gelu(x)
        return nn.Dense(N_CLASSES, name="cls")(x)  # (B, n_slots, N_CLASSES)


class FlagellumHead(nn.Module):
    """Predict flagellum params + variances per slot.

    Output dict per slot:
      points_mean:  (K+1, 2) — attachment + K control points, canonical (y, x) px
      points_log_sigma: (K+1, 2)  — log σ for each coordinate (diagonal Gaussian)
      width_mean, width_log_sigma:      scalars (canonical px)
      amp_mean, amp_log_sigma:          scalars (canonical σ-units)
      polarity_logit:                    binary logit for polarity=darker (+1)
    """
    d_ff: int = 256

    @nn.compact
    def __call__(self, slots: jnp.ndarray) -> dict:
        B, S, D = slots.shape
        n_pts = FLAGELLUM_K + 1
        x = nn.Dense(self.d_ff, name="fc1")(slots)
        x = jax.nn.gelu(x)
        x = nn.Dense(self.d_ff, name="fc2")(x)
        x = jax.nn.gelu(x)

        # Points mean: predict as offsets from a per-slot base position (sigmoid → canvas box)
        pts_mean = nn.Dense(n_pts * 2, name="pts_mean")(x)              # (B, S, 2*(K+1))
        pts_mean = pts_mean.reshape(B, S, n_pts, 2)
        # Rescale to canvas coordinate range — bias to center via sigmoid, then scale
        pts_mean = jax.nn.sigmoid(pts_mean) * jnp.array([CANONICAL_H, CANONICAL_W])

        pts_log_sigma = nn.Dense(n_pts * 2, name="pts_log_sigma")(x)
        pts_log_sigma = pts_log_sigma.reshape(B, S, n_pts, 2)
        # log σ centered at log(20) ≈ 3.0 so at init σ ≈ 20 px, close to typical init errors.
        # Range [-2, 8] → σ ∈ [~0.14, ~2980] px, wide enough to shrink or grow.
        pts_log_sigma = 3.0 + 5.0 * jax.nn.tanh(pts_log_sigma)

        # Auxiliary scalars
        aux = nn.Dense(5, name="aux")(x)                                  # (B, S, 5)
        width_mean = 2.5 + 6.0 * jax.nn.sigmoid(aux[..., 0])               # in [2.5, 8.5]
        width_log_sigma = 0.0 + 3.0 * jax.nn.tanh(aux[..., 1])             # in [-3, 3]
        amp_mean = 2.0 + 15.0 * jax.nn.sigmoid(aux[..., 2])                # in [2, 17]
        amp_log_sigma = 1.0 + 3.0 * jax.nn.tanh(aux[..., 3])               # in [-2, 4]
        polarity_logit = aux[..., 4]

        return dict(
            pts_mean=pts_mean, pts_log_sigma=pts_log_sigma,
            width_mean=width_mean, width_log_sigma=width_log_sigma,
            amp_mean=amp_mean, amp_log_sigma=amp_log_sigma,
            polarity_logit=polarity_logit,
        )


class CellHead(nn.Module):
    """Predict cell body params + variances per slot.

    Output dict:
      cell_center_mean:  (B, S, 2) — canonical (y, x) px
      cell_center_log_sigma: (B, S, 2)
      cell_radius_mean:  (B, S)  canonical px
      cell_radius_log_sigma: (B, S)
      cell_amp_mean:     (B, S)  canonical σ-units
      cell_amp_log_sigma: (B, S)
    """
    d_ff: int = 128

    @nn.compact
    def __call__(self, slots: jnp.ndarray) -> dict:
        B, S, D = slots.shape
        x = nn.Dense(self.d_ff, name="fc1")(slots)
        x = jax.nn.gelu(x)

        center_mean_raw = nn.Dense(2, name="center_mean")(x)
        center_mean = jax.nn.sigmoid(center_mean_raw) * jnp.array([CANONICAL_H, CANONICAL_W])
        center_log_sigma = nn.Dense(2, name="center_log_sigma")(x)
        center_log_sigma = 2.0 + 4.0 * jax.nn.tanh(center_log_sigma)   # in [-2, 6]

        radius_raw = nn.Dense(1, name="radius_mean")(x)[..., 0]
        radius_mean = 8.0 + 70.0 * jax.nn.sigmoid(radius_raw)          # in [8, 78]
        radius_log_sigma = nn.Dense(1, name="radius_log_sigma")(x)[..., 0]
        radius_log_sigma = 1.0 + 3.0 * jax.nn.tanh(radius_log_sigma)    # in [-2, 4]

        amp_raw = nn.Dense(1, name="amp_mean")(x)[..., 0]
        amp_mean = 2.0 + 15.0 * jax.nn.sigmoid(amp_raw)                # in [2, 17]
        amp_log_sigma = nn.Dense(1, name="amp_log_sigma")(x)[..., 0]
        amp_log_sigma = 1.0 + 3.0 * jax.nn.tanh(amp_log_sigma)

        return dict(
            cell_center_mean=center_mean,
            cell_center_log_sigma=center_log_sigma,
            cell_radius_mean=radius_mean,
            cell_radius_log_sigma=radius_log_sigma,
            cell_amp_mean=amp_mean,
            cell_amp_log_sigma=amp_log_sigma,
        )


def sample_cell_from_head(rng: jax.Array, head_out: dict, n_samples: int, temperature: float = 1.0) -> dict:
    """Draw n_samples per slot from the cell head."""
    B, S = head_out["cell_center_mean"].shape[:2]
    keys = jax.random.split(rng, 3)
    c_sigma = jnp.exp(head_out["cell_center_log_sigma"]) * temperature
    c_noise = jax.random.normal(keys[0], (B, S, n_samples, 2))
    center_samples = head_out["cell_center_mean"][:, :, None] + c_sigma[:, :, None] * c_noise
    r_sigma = jnp.exp(head_out["cell_radius_log_sigma"]) * temperature
    r_noise = jax.random.normal(keys[1], (B, S, n_samples))
    radius_samples = head_out["cell_radius_mean"][:, :, None] + r_sigma[:, :, None] * r_noise
    a_sigma = jnp.exp(head_out["cell_amp_log_sigma"]) * temperature
    a_noise = jax.random.normal(keys[2], (B, S, n_samples))
    amp_samples = head_out["cell_amp_mean"][:, :, None] + a_sigma[:, :, None] * a_noise
    return dict(
        center_samples=center_samples,
        radius_samples=radius_samples,
        amp_samples=amp_samples,
    )


def sample_flagellum_from_head(rng: jax.Array, head_out: dict, n_samples: int, temperature: float = 1.0) -> dict:
    """Draw n_samples per slot from the head's predicted Gaussians.

    Returns dict with:
      pts_samples:      (B, S, n_samples, K+1, 2)
      width_samples:    (B, S, n_samples)
      amp_samples:      (B, S, n_samples)
      polarity_samples: (B, S, n_samples)  each ±1
    """
    B, S = head_out["pts_mean"].shape[:2]
    n_pts = head_out["pts_mean"].shape[2]
    keys = jax.random.split(rng, 4)

    pts_sigma = jnp.exp(head_out["pts_log_sigma"]) * temperature
    pts_noise = jax.random.normal(keys[0], (B, S, n_samples, n_pts, 2))
    pts_samples = head_out["pts_mean"][:, :, None] + pts_sigma[:, :, None] * pts_noise

    width_sigma = jnp.exp(head_out["width_log_sigma"]) * temperature
    width_noise = jax.random.normal(keys[1], (B, S, n_samples))
    width_samples = head_out["width_mean"][:, :, None] + width_sigma[:, :, None] * width_noise

    amp_sigma = jnp.exp(head_out["amp_log_sigma"]) * temperature
    amp_noise = jax.random.normal(keys[2], (B, S, n_samples))
    amp_samples = head_out["amp_mean"][:, :, None] + amp_sigma[:, :, None] * amp_noise

    # Polarity: Bernoulli, we sample discretely at inference (positive → +1 = darker).
    p_pos = jax.nn.sigmoid(head_out["polarity_logit"])                  # (B, S)
    u = jax.random.uniform(keys[3], (B, S, n_samples))
    polarity_samples = jnp.where(u < p_pos[:, :, None], 1.0, -1.0)

    return dict(
        pts_samples=pts_samples,
        width_samples=width_samples,
        amp_samples=amp_samples,
        polarity_samples=polarity_samples,
    )


def class_and_flagellum_from_slot(slots: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    """Convenience: apply both heads in one call. Returns (class_logits, flagellum_head_out)."""
    class_logits = ClassHead(name="class_head")(slots)
    flag_out = FlagellumHead(name="flagellum_head")(slots)
    return class_logits, flag_out
