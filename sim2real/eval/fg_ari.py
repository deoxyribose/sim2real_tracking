"""Foreground Adjusted Rand Index (FG-ARI).

Standard object-centric metric (Locatello 2020 "Object-Centric Learning with Slot Attention";
also used in SAVi, STEVE, etc.). Compared to matched-seg-IoU it is:

- Permutation-invariant: no Hungarian, no ambiguity about which pred slot is which.
- Restricted to foreground pixels: ignores the background class, so a model that pushes all
  slots into "background" cannot get a free win.
- Scale-invariant: measures cluster agreement over foreground pixels, not per-pixel overlap.

Definition per frame:
  1. Assign every foreground pixel to a slot cluster by argmax over slot masks. Do this
     independently for pred and GT.
  2. Foreground = pixels where the sum of GT alive-slot masks > `fg_thresh`.
  3. Compute Adjusted Rand Index (ARI) on the two per-pixel clusterings, restricted to
     foreground pixels.
  4. Return the mean ARI over frames (skipping frames with < 2 foreground pixels or < 2
     unique GT clusters, where ARI is undefined).

References:
  - Hubert & Arabie 1985, "Comparing Partitions" — ARI definition.
  - Locatello et al. 2020 — first use in object-centric ML with FG restriction.
"""

from __future__ import annotations

import numpy as np


def _ari(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Adjusted Rand Index between two 1-D integer label arrays. Returns NaN when either
    partition is trivial (0 or 1 cluster) or when the arrays are empty."""
    if labels_true.size == 0:
        return float("nan")
    # Build contingency table via unique inverse remap so cluster ids need not be dense.
    _, t = np.unique(labels_true, return_inverse=True)
    _, p = np.unique(labels_pred, return_inverse=True)
    n_t = int(t.max()) + 1
    n_p = int(p.max()) + 1
    if n_t < 2 and n_p < 2:
        return float("nan")
    C = np.zeros((n_t, n_p), dtype=np.int64)
    np.add.at(C, (t, p), 1)

    def comb2(x):
        return x * (x - 1) // 2

    sum_comb_c = comb2(C).sum()
    a = C.sum(axis=1)
    b = C.sum(axis=0)
    sum_comb_a = comb2(a).sum()
    sum_comb_b = comb2(b).sum()
    n = int(labels_true.size)
    total = comb2(n)
    expected = (sum_comb_a * sum_comb_b) / total if total > 0 else 0.0
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_index - expected
    if denom == 0:
        return float("nan")
    return float((sum_comb_c - expected) / denom)


def fg_ari_frame(pred_masks_frame: np.ndarray, gt_masks_frame: np.ndarray,
                 gt_pres_frame: np.ndarray, fg_thresh: float = 0.5) -> float:
    """FG-ARI on a single frame.

    Args:
      pred_masks_frame: (N, H, W) — predicted slot masks, values in [0, 1].
      gt_masks_frame:   (N, H, W) — GT slot masks, values in [0, 1].
      gt_pres_frame:    (N,)       — 1 = alive GT slot, 0 = padding.
      fg_thresh:        pixel is foreground if the sum of alive-GT masks exceeds this.
    """
    alive = gt_pres_frame > 0.5                                              # (N,)
    gt_alive_masks = gt_masks_frame * alive[:, None, None]                   # (N, H, W)
    fg_mass = gt_alive_masks.sum(axis=0)                                     # (H, W)
    fg = fg_mass > fg_thresh                                                  # bool (H, W)
    if fg.sum() < 2:
        return float("nan")

    # Per-pixel cluster ids on foreground pixels only.
    # Pred: argmax across ALL pred slots (background pixels excluded by `fg`, so no need to gate).
    pred_labels = pred_masks_frame.argmax(axis=0)[fg]                        # (P,)
    # GT: argmax across the ALIVE GT slots only — inject -inf on dead slots so they never win.
    gt_scores = gt_masks_frame.copy()
    gt_scores[~alive] = -np.inf
    gt_labels = gt_scores.argmax(axis=0)[fg]                                 # (P,)
    return _ari(gt_labels, pred_labels)


def fg_ari(pred_masks: np.ndarray, gt_masks: np.ndarray, gt_pres: np.ndarray,
           fg_thresh: float = 0.5) -> float:
    """FG-ARI averaged over frames of one video (undefined frames skipped).

    Args:
      pred_masks: (T, N, H, W)
      gt_masks:   (T, N, H, W)
      gt_pres:    (T, N)
    """
    T = pred_masks.shape[0]
    vals = []
    for t in range(T):
        v = fg_ari_frame(pred_masks[t], gt_masks[t], gt_pres[t], fg_thresh)
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")
