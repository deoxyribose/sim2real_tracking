"""Ordered knot-to-knot L2 metric — an alternative to symmetric Chamfer that
does NOT let snaking predictions win.

For pred (P points) vs GT (G points):
  1. Resample both to a common K = max(min(P, G, K_max), K_min) points
     (uniform arc-length reparameterisation).
  2. Compute L2 distance between knot i of pred and knot i of GT.
  3. Also compute the flipped version (pred reversed).
  4. Return min of (mean(L2), mean(L2_flipped)).

This penalises:
  - Over-extension (rollout past GT end) — those knots pair with nothing near.
  - Under-coverage — GT has no matching pred knot.
  - Snaking / S-curves — knot-i distances balloon.

Chamfer's failure mode (snaking prediction registers as low error) is fixed
because there's no min-over-all-points: knot i pairs with knot i, period.
"""
from __future__ import annotations

import numpy as np


def resample_polyline(polyline: np.ndarray, k: int) -> np.ndarray:
    """Reparameterise a (N, 2) polyline to k evenly-arc-spaced points."""
    if len(polyline) < 2:
        return np.repeat(polyline[:1], k, axis=0)
    segs = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(segs)])
    if cum[-1] < 1e-9:
        return np.repeat(polyline[:1], k, axis=0)
    tgt = np.linspace(0, cum[-1], k)
    x = np.interp(tgt, cum, polyline[:, 1])
    y = np.interp(tgt, cum, polyline[:, 0])
    return np.stack([y, x], axis=-1)


def ordered_l2(pred: np.ndarray, gt: np.ndarray, k: int = 25) -> float:
    """Mean L2 distance between knot i of pred and knot i of gt, both
    resampled to k points. Takes min over pred vs pred[::-1] to allow
    reverse direction."""
    if len(pred) < 2 or len(gt) < 2:
        return float("inf")
    p = resample_polyline(pred, k)
    g = resample_polyline(gt, k)
    d1 = float(np.linalg.norm(p - g, axis=1).mean())
    d2 = float(np.linalg.norm(p[::-1] - g, axis=1).mean())
    return min(d1, d2)


def hausdorff(pred: np.ndarray, gt: np.ndarray) -> float:
    """Symmetric Hausdorff (max of directed max-of-mins)."""
    if len(pred) == 0 or len(gt) == 0:
        return float("inf")
    d = np.linalg.norm(pred[:, None] - gt[None], axis=-1)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))
