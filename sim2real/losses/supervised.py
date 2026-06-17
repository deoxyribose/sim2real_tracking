"""Supervised losses applied after Hungarian matching:

- L_where: MSE on z_where (only for alive GT slots).
- L_pres: two-sided BCE-from-logits (numerically stable at saturation; works on the soft pres
  logit rather than the Gumbel-ST hard output).
- L_mask: BCE + soft Dice on predicted vs GT masks, **including padding slots** (whose target
  becomes all-zeros) so the seg head learns to suppress dead-slot output.

All operate on (T, N, ...) tensors already aligned by the matching perm.
"""

from __future__ import annotations

import jax.nn
import jax.numpy as jnp

Array = jnp.ndarray


def masked_mse(pred: Array, gt: Array, alive: Array) -> Array:
    """`alive` is (T, N) — 1 for valid GT slots. Squared error averaged over the trailing dims and
    summed over alive entries, then normalized by total alive count."""
    diff_sq = jnp.sum((pred - gt) ** 2, axis=-1)                                     # (T, N)
    num = jnp.sum(diff_sq * alive)
    den = jnp.sum(alive) + 1e-6
    return num / den


def bce_from_logits(logit: Array, gt_pres: Array) -> Array:
    """Two-sided BCE computed directly on logits — numerically stable at saturation.

    L = mean( logsumexp(0, -z) + (1 - gt) · z ) i.e. mean over (T, N) of:
        log(1 + exp(z)) - gt · z
    """
    return jnp.mean(jax.nn.softplus(logit) - gt_pres * logit)


def bce_one_sided(pred_pres: Array, gt_pres: Array, eps: float = 1e-6) -> Array:
    """Deprecated — kept for the adapt loop. Use bce_from_logits in pretrain."""
    pred = jnp.clip(pred_pres, eps, 1.0 - eps)
    return -jnp.mean((1.0 - gt_pres) * jnp.log(1.0 - pred))


def bce_two_sided(pred_pres: Array, gt_pres: Array, eps: float = 1e-6) -> Array:
    pred = jnp.clip(pred_pres, eps, 1.0 - eps)
    return -jnp.mean(gt_pres * jnp.log(pred) + (1.0 - gt_pres) * jnp.log(1.0 - pred))


def mask_loss(pred_mask: Array, gt_mask: Array, alive: Array, dice_weight: float = 1.0) -> Array:
    """BCE + soft-Dice over ALL slots.

    Dead slots (alive=0) get their GT mask zeroed so the seg head is supervised to output empty
    masks for padding. This prevents the "uniform red square over the canvas" failure mode where
    padding slots' masks drift at sigmoid(init)≈0.5.

    Args:
      pred_mask: (T, N, H, W) in [0,1].
      gt_mask:   (T, N, H, W) in [0,1].
      alive:     (T, N) in {0, 1} — GT z_pres.
    """
    eps = 1e-6
    gt_effective = gt_mask * alive[..., None, None]                                   # zero out dead
    p = jnp.clip(pred_mask, eps, 1.0 - eps)
    bce = -(gt_effective * jnp.log(p) + (1.0 - gt_effective) * jnp.log(1.0 - p))      # (T,N,H,W)
    bce_per_slot = jnp.mean(bce, axis=(-1, -2))                                       # (T, N)

    # Soft Dice — alive slots use real GT; dead slots get a small offset to keep dice well-defined
    # (we add eps to denom so 0/0 → 1−0 = 1, then take 1 − 2·0/eps = 1, then dice_weight·1 is
    # a constant offset).
    inter = jnp.sum(p * gt_effective, axis=(-1, -2))
    denom = jnp.sum(p + gt_effective, axis=(-1, -2)) + eps
    dice = 1.0 - 2.0 * inter / denom                                                  # (T, N)

    per_slot = bce_per_slot + dice_weight * dice
    return jnp.mean(per_slot)
