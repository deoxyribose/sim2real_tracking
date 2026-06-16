"""Supervised losses applied after Hungarian matching:

- L_where: MSE on z_where (only for alive GT slots).
- L_pres: one-sided BCE (only penalize false-positive presence; let recon pull z_pres up).
- L_mask: BCE + soft Dice on predicted vs GT masks.

All operate on (T, N, ...) tensors already aligned by the matching perm.
"""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray


def masked_mse(pred: Array, gt: Array, alive: Array) -> Array:
    """`alive` is (T, N) — 1 for valid GT slots. Squared error averaged over the trailing dims and
    summed over alive entries, then normalized by total alive count."""
    diff_sq = jnp.sum((pred - gt) ** 2, axis=-1)                                     # (T, N)
    num = jnp.sum(diff_sq * alive)
    den = jnp.sum(alive) + 1e-6
    return num / den


def bce_one_sided(pred_pres: Array, gt_pres: Array, eps: float = 1e-6) -> Array:
    """One-sided BCE: only penalize false-positive presence (z_pres > 0 when gt_pres = 0).

    L = -mean( (1 - gt) * log(1 - pred) ).
    """
    pred = jnp.clip(pred_pres, eps, 1.0 - eps)
    return -jnp.mean((1.0 - gt_pres) * jnp.log(1.0 - pred))


def bce_two_sided(pred_pres: Array, gt_pres: Array, eps: float = 1e-6) -> Array:
    pred = jnp.clip(pred_pres, eps, 1.0 - eps)
    return -jnp.mean(gt_pres * jnp.log(pred) + (1.0 - gt_pres) * jnp.log(1.0 - pred))


def mask_loss(pred_mask: Array, gt_mask: Array, alive: Array, dice_weight: float = 1.0) -> Array:
    """Mean BCE + soft-Dice over alive slots.

    Args:
      pred_mask: (T, N, H, W) in [0,1].
      gt_mask:   (T, N, H, W) in [0,1].
      alive:     (T, N)
    """
    eps = 1e-6
    p = jnp.clip(pred_mask, eps, 1.0 - eps)
    bce = -(gt_mask * jnp.log(p) + (1.0 - gt_mask) * jnp.log(1.0 - p))                # (T,N,H,W)
    bce_per_slot = jnp.mean(bce, axis=(-1, -2))                                       # (T, N)

    # Soft Dice per slot
    inter = jnp.sum(p * gt_mask, axis=(-1, -2))
    denom = jnp.sum(p + gt_mask, axis=(-1, -2)) + eps
    dice = 1.0 - 2.0 * inter / denom                                                  # (T, N)

    per_slot = bce_per_slot + dice_weight * dice
    num = jnp.sum(per_slot * alive)
    den = jnp.sum(alive) + 1e-6
    return num / den
