"""Per-slot segmentation loss (used by the dedicated seg head)."""

from __future__ import annotations

import jax.numpy as jnp

from sim2real.losses.supervised import mask_loss


def seg_loss(pred_mask, gt_mask, alive, dice_weight: float = 1.0):
    return mask_loss(pred_mask, gt_mask, alive, dice_weight)
