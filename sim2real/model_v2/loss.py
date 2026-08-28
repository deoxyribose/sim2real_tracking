"""DETR-style Hungarian matching + NLL loss for the slot model.

Match K predicted slots to G ground-truth flagella per batch element (G varies 0-2).
Cost = class_cost (want class=flagellum) + position_cost (attachment L2) + curve_chamfer.

Matched slots get NLL under their predicted Gaussians + class cross-entropy toward
`flagellum`. Unmatched slots get class cross-entropy toward `empty`.

The Hungarian solver runs on host (scipy). We wrap it in `jax.pure_callback` so it
can be called inside JIT'd training steps, or the whole loss can be computed outside
JIT — both paths are supported. Matches are `stop_gradient`.
"""
from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..data import CLASS_EMPTY, CLASS_FLAGELLUM, FLAGELLUM_K, N_CLASSES


# ------------------------------- GT packing --------------------------------

MAX_GT = 2  # max flagella per clip (matches sim's 2-flagella cap)


def pack_gt_batch(scenes) -> dict:
    """Batch a list of SceneLatents into padded numpy arrays.

    Returns dict:
      gt_pts:      (B, MAX_GT, K+1, 2)  float32
      gt_width:    (B, MAX_GT)          float32
      gt_amp:      (B, MAX_GT)          float32
      gt_polarity: (B, MAX_GT)          float32   ∈ {-1, +1}
      gt_mask:     (B, MAX_GT)          float32   ∈ {0, 1}  (1 = real GT, 0 = padding)
    """
    B = len(scenes)
    K1 = FLAGELLUM_K + 1
    gt_pts = np.zeros((B, MAX_GT, K1, 2), dtype=np.float32)
    gt_width = np.zeros((B, MAX_GT), dtype=np.float32)
    gt_amp = np.zeros((B, MAX_GT), dtype=np.float32)
    gt_polarity = np.zeros((B, MAX_GT), dtype=np.float32)
    gt_mask = np.zeros((B, MAX_GT), dtype=np.float32)
    for b, scene in enumerate(scenes):
        for g, flg in enumerate(scene.flagella[:MAX_GT]):
            gt_pts[b, g] = flg.all_points()
            gt_width[b, g] = flg.width_px
            gt_amp[b, g] = flg.amplitude_sigma
            gt_polarity[b, g] = float(flg.polarity)
            gt_mask[b, g] = 1.0
    return dict(gt_pts=gt_pts, gt_width=gt_width, gt_amp=gt_amp,
                gt_polarity=gt_polarity, gt_mask=gt_mask)


# ------------------------------- Cost matrices --------------------------------

def _curve_chamfer_np(pred_pts: np.ndarray, gt_pts: np.ndarray) -> np.ndarray:
    """Chamfer distance for two point sets. Shapes:
        pred_pts: (S, K+1, 2)
        gt_pts:   (G, K+1, 2)
    Returns:      (S, G) float32
    """
    # Pairwise dists (S, G, K+1, K+1)
    d = np.linalg.norm(pred_pts[:, None, :, None] - gt_pts[None, :, None, :], axis=-1)
    p_to_g = d.min(axis=-1).mean(axis=-1)   # (S, G)
    g_to_p = d.min(axis=-2).mean(axis=-1)   # (S, G)
    return 0.5 * (p_to_g + g_to_p)


def hungarian_match(
    class_logits_np: np.ndarray,   # (B, S, N_CLASSES)
    pred_pts_np: np.ndarray,       # (B, S, K+1, 2)
    gt_pts_np: np.ndarray,         # (B, MAX_GT, K+1, 2)
    gt_mask_np: np.ndarray,        # (B, MAX_GT)  1 = real, 0 = padding
    cost_class_weight: float = 1.0,
    cost_pos_weight: float = 1.0,
    cost_curve_weight: float = 5.0,
) -> np.ndarray:
    """Return match matrix (B, S) with the GT index each slot is matched to, or -1 if unmatched.

    Only real (mask=1) GT positions are considered targets. Standard DETR matching:
    for each batch element separately, compute a (S, G_real) cost matrix, run
    linear_sum_assignment, assign match[slot] = matched_gt or -1.
    """
    B, S, K1, _ = pred_pts_np.shape
    G = gt_pts_np.shape[1]
    match = np.full((B, S), -1, dtype=np.int32)
    # For softmax class probs (approx — sigmoid gives per-class, but use softmax over N_CLASSES here)
    class_probs = np.exp(class_logits_np - class_logits_np.max(-1, keepdims=True))
    class_probs = class_probs / class_probs.sum(-1, keepdims=True)
    p_flag = class_probs[..., CLASS_FLAGELLUM]  # (B, S)  higher = better for flagellum
    for b in range(B):
        real = np.where(gt_mask_np[b] > 0.5)[0]
        if len(real) == 0:
            continue
        gt_p = gt_pts_np[b, real]     # (G_real, K+1, 2)
        pr_p = pred_pts_np[b]          # (S, K+1, 2)
        # Position cost: attachment L2
        pos_cost = np.linalg.norm(pr_p[:, 0][:, None] - gt_p[:, 0][None], axis=-1)  # (S, G_real)
        # Curve cost: chamfer
        curve_cost = _curve_chamfer_np(pr_p, gt_p)  # (S, G_real)
        # Class cost: -log p(flagellum) — lower is better
        class_cost = -np.log(np.clip(p_flag[b], 1e-6, 1.0))[:, None]  # (S, 1) broadcast
        cost = cost_class_weight * class_cost + cost_pos_weight * pos_cost + cost_curve_weight * curve_cost
        row, col = linear_sum_assignment(cost)
        for r, c in zip(row, col):
            match[b, r] = real[c]
    return match


# ------------------------------- Loss primitives --------------------------------

def _gaussian_nll(x: jnp.ndarray, mean: jnp.ndarray, log_sigma: jnp.ndarray) -> jnp.ndarray:
    """Element-wise diagonal Gaussian NLL."""
    sigma = jnp.exp(log_sigma)
    return log_sigma + 0.5 * ((x - mean) / sigma) ** 2 + 0.5 * jnp.log(2 * jnp.pi)


def compute_loss(
    outputs: dict,
    gt: dict,
    match: jnp.ndarray,                 # (B, S) int32, -1 = unmatched
    class_weight: float = 1.0,
    pts_weight: float = 1.0,
    width_weight: float = 0.5,
    amp_weight: float = 0.5,
    polarity_weight: float = 0.5,
) -> dict:
    """All-in-one loss.

    outputs: model.apply outputs (jax arrays)
    gt: batched GT dict from pack_gt_batch (jax arrays)
    match: (B, S) int32 — -1 means unmatched (supervised to class=EMPTY)

    Returns dict with 'total' + component breakdown.
    """
    B, S = outputs["class_logits"].shape[:2]

    # ----------------- Class loss -----------------
    # Target class per slot: flagellum where matched, empty where unmatched.
    matched = match >= 0                                                    # (B, S) bool
    target_class = jnp.where(matched, CLASS_FLAGELLUM, CLASS_EMPTY)         # (B, S) int
    logp = jax.nn.log_softmax(outputs["class_logits"], axis=-1)             # (B, S, N)
    class_ll = jnp.take_along_axis(logp, target_class[..., None], axis=-1)[..., 0]  # (B, S)
    loss_class = -class_ll.mean()

    # ----------------- Matched-only losses -----------------
    # Gather GT for each slot; use safe_idx to avoid OOB, then mask.
    safe_gt_idx = jnp.clip(match, 0, gt["gt_pts"].shape[1] - 1)             # (B, S)
    b_idx = jnp.arange(B)[:, None]                                           # (B, 1)

    gt_pts_slot = gt["gt_pts"][b_idx, safe_gt_idx]                          # (B, S, K+1, 2)
    gt_width_slot = gt["gt_width"][b_idx, safe_gt_idx]                      # (B, S)
    gt_amp_slot = gt["gt_amp"][b_idx, safe_gt_idx]                          # (B, S)
    gt_pol_slot = gt["gt_polarity"][b_idx, safe_gt_idx]                     # (B, S) ±1

    pts_nll = _gaussian_nll(gt_pts_slot, outputs["pts_mean"], outputs["pts_log_sigma"]).sum((-1, -2))  # (B, S)
    width_nll = _gaussian_nll(gt_width_slot, outputs["width_mean"], outputs["width_log_sigma"])
    amp_nll = _gaussian_nll(gt_amp_slot, outputs["amp_mean"], outputs["amp_log_sigma"])
    # Polarity as BCE: target = 1 if darker (+1), 0 if brighter (-1)
    pol_target = (gt_pol_slot > 0).astype(jnp.float32)
    pol_ce = -(pol_target * jax.nn.log_sigmoid(outputs["polarity_logit"])
               + (1 - pol_target) * jax.nn.log_sigmoid(-outputs["polarity_logit"]))

    matched_f = matched.astype(jnp.float32)
    n_matched = matched_f.sum() + 1e-6
    loss_pts = (pts_nll * matched_f).sum() / n_matched
    loss_width = (width_nll * matched_f).sum() / n_matched
    loss_amp = (amp_nll * matched_f).sum() / n_matched
    loss_pol = (pol_ce * matched_f).sum() / n_matched

    total = (class_weight * loss_class
             + pts_weight * loss_pts
             + width_weight * loss_width
             + amp_weight * loss_amp
             + polarity_weight * loss_pol)

    return dict(
        total=total, class_=loss_class, pts=loss_pts, width=loss_width,
        amp=loss_amp, polarity=loss_pol,
        n_matched=n_matched,
    )
