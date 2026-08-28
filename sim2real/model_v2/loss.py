"""DETR-style Hungarian matching + NLL loss for the slot model (v1 — cells + flagella).

Two separate Hungarian matches per batch element: one for flagella, one for cells.
Slots not matched to any GT are supervised to CLASS_EMPTY.

The Hungarian solver runs on host (scipy). Matches are `stop_gradient`.
"""
from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..data import CLASS_EMPTY, CLASS_CELL, CLASS_FLAGELLUM, FLAGELLUM_K


# ------------------------------- GT packing --------------------------------

MAX_GT_FLAG = 4    # up to 2 cells × 2 flagella each
MAX_GT_CELL = 2


def pack_gt_batch(scenes) -> dict:
    """Batch a list of SceneLatents into padded numpy arrays."""
    B = len(scenes)
    K1 = FLAGELLUM_K + 1
    gt_pts = np.zeros((B, MAX_GT_FLAG, K1, 2), dtype=np.float32)
    gt_width = np.zeros((B, MAX_GT_FLAG), dtype=np.float32)
    gt_amp = np.zeros((B, MAX_GT_FLAG), dtype=np.float32)
    gt_polarity = np.zeros((B, MAX_GT_FLAG), dtype=np.float32)
    gt_mask = np.zeros((B, MAX_GT_FLAG), dtype=np.float32)

    gt_cell_center = np.zeros((B, MAX_GT_CELL, 2), dtype=np.float32)
    gt_cell_radius = np.zeros((B, MAX_GT_CELL), dtype=np.float32)
    gt_cell_amp = np.zeros((B, MAX_GT_CELL), dtype=np.float32)
    gt_cell_mask = np.zeros((B, MAX_GT_CELL), dtype=np.float32)

    for b, scene in enumerate(scenes):
        for g, flg in enumerate(scene.flagella[:MAX_GT_FLAG]):
            gt_pts[b, g] = flg.all_points()
            gt_width[b, g] = flg.width_px
            gt_amp[b, g] = flg.amplitude_sigma
            gt_polarity[b, g] = float(flg.polarity)
            gt_mask[b, g] = 1.0
        for g, cell in enumerate(scene.cells[:MAX_GT_CELL]):
            gt_cell_center[b, g] = cell.center
            gt_cell_radius[b, g] = cell.radius_px
            gt_cell_amp[b, g] = cell.amplitude_sigma
            gt_cell_mask[b, g] = 1.0

    return dict(
        gt_pts=gt_pts, gt_width=gt_width, gt_amp=gt_amp,
        gt_polarity=gt_polarity, gt_mask=gt_mask,
        gt_cell_center=gt_cell_center, gt_cell_radius=gt_cell_radius,
        gt_cell_amp=gt_cell_amp, gt_cell_mask=gt_cell_mask,
    )


# ------------------------------- Cost matrices --------------------------------

def _curve_chamfer_np(pred_pts: np.ndarray, gt_pts: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(pred_pts[:, None, :, None] - gt_pts[None, :, None, :], axis=-1)
    p_to_g = d.min(axis=-1).mean(axis=-1)
    g_to_p = d.min(axis=-2).mean(axis=-1)
    return 0.5 * (p_to_g + g_to_p)


def hungarian_match_flagella(
    class_logits_np: np.ndarray,
    pred_pts_np: np.ndarray,
    gt_pts_np: np.ndarray,
    gt_mask_np: np.ndarray,
    slot_forbidden_np: np.ndarray,
    cost_class_weight: float = 1.0,
    cost_pos_weight: float = 1.0,
    cost_curve_weight: float = 5.0,
) -> np.ndarray:
    """Match slots→flagellum GT. slot_forbidden_np: (B, S) bool; True = slot already used by cell match."""
    B, S = pred_pts_np.shape[:2]
    match = np.full((B, S), -1, dtype=np.int32)
    class_probs = np.exp(class_logits_np - class_logits_np.max(-1, keepdims=True))
    class_probs = class_probs / class_probs.sum(-1, keepdims=True)
    p_flag = class_probs[..., CLASS_FLAGELLUM]
    for b in range(B):
        real = np.where(gt_mask_np[b] > 0.5)[0]
        if len(real) == 0:
            continue
        # Restrict to slots not forbidden
        avail = np.where(~slot_forbidden_np[b])[0]
        if len(avail) < len(real):
            avail = np.arange(S)  # fallback — allow overlap if we can't otherwise satisfy demand
        gt_p = gt_pts_np[b, real]
        pr_p = pred_pts_np[b, avail]
        pos_cost = np.linalg.norm(pr_p[:, 0][:, None] - gt_p[:, 0][None], axis=-1)
        curve_cost = _curve_chamfer_np(pr_p, gt_p)
        class_cost = -np.log(np.clip(p_flag[b, avail], 1e-6, 1.0))[:, None]
        cost = cost_class_weight * class_cost + cost_pos_weight * pos_cost + cost_curve_weight * curve_cost
        row, col = linear_sum_assignment(cost)
        for r, c in zip(row, col):
            match[b, avail[r]] = real[c]
    return match


def hungarian_match_cells(
    class_logits_np: np.ndarray,
    pred_center_np: np.ndarray,   # (B, S, 2)
    pred_radius_np: np.ndarray,   # (B, S)
    gt_center_np: np.ndarray,     # (B, MAX_GT_CELL, 2)
    gt_radius_np: np.ndarray,     # (B, MAX_GT_CELL)
    gt_mask_np: np.ndarray,       # (B, MAX_GT_CELL)
    cost_class_weight: float = 1.0,
    cost_pos_weight: float = 1.0,
    cost_radius_weight: float = 0.5,
) -> np.ndarray:
    B, S = pred_center_np.shape[:2]
    match = np.full((B, S), -1, dtype=np.int32)
    class_probs = np.exp(class_logits_np - class_logits_np.max(-1, keepdims=True))
    class_probs = class_probs / class_probs.sum(-1, keepdims=True)
    p_cell = class_probs[..., CLASS_CELL]
    for b in range(B):
        real = np.where(gt_mask_np[b] > 0.5)[0]
        if len(real) == 0:
            continue
        gt_c = gt_center_np[b, real]        # (G, 2)
        gt_r = gt_radius_np[b, real]        # (G,)
        pr_c = pred_center_np[b]            # (S, 2)
        pr_r = pred_radius_np[b]            # (S,)
        pos_cost = np.linalg.norm(pr_c[:, None] - gt_c[None], axis=-1)  # (S, G)
        rad_cost = np.abs(pr_r[:, None] - gt_r[None])                    # (S, G)
        class_cost = -np.log(np.clip(p_cell[b], 1e-6, 1.0))[:, None]
        cost = cost_class_weight * class_cost + cost_pos_weight * pos_cost + cost_radius_weight * rad_cost
        row, col = linear_sum_assignment(cost)
        for r, c in zip(row, col):
            match[b, r] = real[c]
    return match


# ------------------------------- Loss primitives --------------------------------

def _gaussian_nll(x: jnp.ndarray, mean: jnp.ndarray, log_sigma: jnp.ndarray) -> jnp.ndarray:
    sigma = jnp.exp(log_sigma)
    return log_sigma + 0.5 * ((x - mean) / sigma) ** 2 + 0.5 * jnp.log(2 * jnp.pi)


def compute_loss(
    outputs: dict,
    gt: dict,
    match_flag: jnp.ndarray,    # (B, S) int32 → -1 or GT flag idx
    match_cell: jnp.ndarray,    # (B, S) int32 → -1 or GT cell idx
    class_weight: float = 1.0,
    pts_weight: float = 1.0,
    width_weight: float = 0.5,
    amp_weight: float = 0.5,
    polarity_weight: float = 0.5,
    cell_pos_weight: float = 1.0,
    cell_radius_weight: float = 0.5,
    cell_amp_weight: float = 0.5,
) -> dict:
    B, S = outputs["class_logits"].shape[:2]

    # ---- Class target: FLAG if matched to flag, CELL if matched to cell, EMPTY otherwise ----
    is_flag = match_flag >= 0
    is_cell = match_cell >= 0
    target_class = jnp.where(is_flag, CLASS_FLAGELLUM,
                             jnp.where(is_cell, CLASS_CELL, CLASS_EMPTY))
    logp = jax.nn.log_softmax(outputs["class_logits"], axis=-1)
    class_ll = jnp.take_along_axis(logp, target_class[..., None], axis=-1)[..., 0]
    loss_class = -class_ll.mean()

    # ---- Flagellum losses (only on flag-matched slots) ----
    safe_flag_idx = jnp.clip(match_flag, 0, gt["gt_pts"].shape[1] - 1)
    b_idx = jnp.arange(B)[:, None]
    gt_pts_slot = gt["gt_pts"][b_idx, safe_flag_idx]
    gt_width_slot = gt["gt_width"][b_idx, safe_flag_idx]
    gt_amp_slot = gt["gt_amp"][b_idx, safe_flag_idx]
    gt_pol_slot = gt["gt_polarity"][b_idx, safe_flag_idx]

    pts_nll = _gaussian_nll(gt_pts_slot, outputs["pts_mean"], outputs["pts_log_sigma"]).sum((-1, -2))
    width_nll = _gaussian_nll(gt_width_slot, outputs["width_mean"], outputs["width_log_sigma"])
    amp_nll = _gaussian_nll(gt_amp_slot, outputs["amp_mean"], outputs["amp_log_sigma"])
    pol_target = (gt_pol_slot > 0).astype(jnp.float32)
    pol_ce = -(pol_target * jax.nn.log_sigmoid(outputs["polarity_logit"])
               + (1 - pol_target) * jax.nn.log_sigmoid(-outputs["polarity_logit"]))

    flag_mask_f = is_flag.astype(jnp.float32)
    n_flag = flag_mask_f.sum() + 1e-6
    loss_pts = (pts_nll * flag_mask_f).sum() / n_flag
    loss_width = (width_nll * flag_mask_f).sum() / n_flag
    loss_amp = (amp_nll * flag_mask_f).sum() / n_flag
    loss_pol = (pol_ce * flag_mask_f).sum() / n_flag

    # ---- Cell losses (only on cell-matched slots) ----
    safe_cell_idx = jnp.clip(match_cell, 0, gt["gt_cell_center"].shape[1] - 1)
    gt_cc = gt["gt_cell_center"][b_idx, safe_cell_idx]        # (B, S, 2)
    gt_cr = gt["gt_cell_radius"][b_idx, safe_cell_idx]        # (B, S)
    gt_ca = gt["gt_cell_amp"][b_idx, safe_cell_idx]           # (B, S)

    cell_pos_nll = _gaussian_nll(gt_cc, outputs["cell_center_mean"], outputs["cell_center_log_sigma"]).sum(-1)
    cell_rad_nll = _gaussian_nll(gt_cr, outputs["cell_radius_mean"], outputs["cell_radius_log_sigma"])
    cell_amp_nll = _gaussian_nll(gt_ca, outputs["cell_amp_mean"], outputs["cell_amp_log_sigma"])

    cell_mask_f = is_cell.astype(jnp.float32)
    n_cell = cell_mask_f.sum() + 1e-6
    loss_cell_pos = (cell_pos_nll * cell_mask_f).sum() / n_cell
    loss_cell_rad = (cell_rad_nll * cell_mask_f).sum() / n_cell
    loss_cell_amp = (cell_amp_nll * cell_mask_f).sum() / n_cell

    total = (class_weight * loss_class
             + pts_weight * loss_pts
             + width_weight * loss_width
             + amp_weight * loss_amp
             + polarity_weight * loss_pol
             + cell_pos_weight * loss_cell_pos
             + cell_radius_weight * loss_cell_rad
             + cell_amp_weight * loss_cell_amp)

    return dict(
        total=total,
        class_=loss_class,
        pts=loss_pts, width=loss_width, amp=loss_amp, polarity=loss_pol,
        cell_pos=loss_cell_pos, cell_rad=loss_cell_rad, cell_amp=loss_cell_amp,
        n_matched_flag=n_flag, n_matched_cell=n_cell,
    )


# ---- Convenience: run both matches (used by training loop) ----
def run_matches(outputs_np: dict, gt_np: dict) -> tuple[np.ndarray, np.ndarray]:
    """Run both Hungarian matches on host. Returns (match_flag, match_cell)."""
    # Cell match first
    match_cell = hungarian_match_cells(
        outputs_np["class_logits"],
        outputs_np["cell_center_mean"],
        outputs_np["cell_radius_mean"],
        gt_np["gt_cell_center"], gt_np["gt_cell_radius"], gt_np["gt_cell_mask"],
    )
    # Flagellum match: forbid slots already used by cells
    slot_forbidden = match_cell >= 0
    match_flag = hungarian_match_flagella(
        outputs_np["class_logits"],
        outputs_np["pts_mean"],
        gt_np["gt_pts"], gt_np["gt_mask"],
        slot_forbidden_np=slot_forbidden,
    )
    return match_flag, match_cell
