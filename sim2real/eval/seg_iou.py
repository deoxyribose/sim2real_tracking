"""Matched per-frame segmentation IoU."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sim2real.losses.matching import hungarian, build_cost_zwhere


def soft_iou(pred_mask: jnp.ndarray, gt_mask: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """`pred_mask`, `gt_mask` shape (..., H, W). Returns IoU per leading entry."""
    inter = jnp.sum(pred_mask * gt_mask, axis=(-1, -2))
    union = jnp.sum(pred_mask + gt_mask - pred_mask * gt_mask, axis=(-1, -2))
    return inter / (union + eps)


def matched_seg_iou(pred_zwhere, pred_masks, gt_zwhere, gt_masks, gt_pres,
                    pred_pres=None, hard_pres_gate: bool = False):
    """Hungarian-match predicted slots to GT slots per frame using z_where, then average IoU
    over alive GT slots and frames.

    Inputs:
      pred_zwhere: (T, N, 3)
      pred_masks:  (T, N, H, W)
      gt_zwhere:   (T, N, 3)
      gt_masks:    (T, N, H, W)
      gt_pres:     (T, N)
      pred_pres:   (T, N) optional — required if hard_pres_gate=True.
      hard_pres_gate: forbid dead-pred slots from winning alive-GT columns.
    """
    def per_frame(pred_zw, pred_m, gt_zw, gt_m, gt_p, pred_p):
        cost = build_cost_zwhere(pred_zw, gt_zw, gt_p, pred_pres=pred_p,
                                 hard_pres_gate=hard_pres_gate)
        perm = hungarian(cost)
        aligned = pred_m[perm]
        ious = soft_iou(aligned, gt_m)                                        # (N,)
        return jnp.sum(ious * gt_p) / (jnp.sum(gt_p) + 1e-6)

    if pred_pres is None:
        pred_pres_arg = jnp.zeros_like(gt_pres)  # unused when hard_pres_gate=False
    else:
        pred_pres_arg = pred_pres
    per_frame_iou = jax.vmap(per_frame)(
        pred_zwhere, pred_masks, gt_zwhere, gt_masks, gt_pres, pred_pres_arg
    )
    return jnp.mean(per_frame_iou)
