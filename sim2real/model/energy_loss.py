"""Energy-score training loss for the grid-based flagellum predictor.

Given two independent noise draws z, z' the model produces two grids of
predictions X, X'. Ground truth is a set of GT skeletons y_j. We optimize:

    L = E_z ‖X − y‖  −  β · ½ E_{z,z'} ‖X − X'‖  +  λ · score_loss

The first term pulls predictions toward the closest GT (accuracy). The
second REWARDS spread between two samples (calibrated diversity) — β<1 so
the term can't dominate and collapse the model. β=0 reduces to standard
supervised learning; β=1 is textbook energy score. We start at β=0.5.

The score head is trained separately with an MSE on the deeptangle-style
"confidence target": s_target = exp(−d² / σ²) where d is the L2 curve
distance to the nearest GT.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from sim2real.model.unet_energy import UNetConfig, decode_curves, unpack_pred

Array = jnp.ndarray


@dataclass(frozen=True)
class EnergyLossConfig:
    beta: float = 0.05          # weight on the diversity (spread) term
    # Cap diversity at `div_ceiling_k · accuracy` so the model can't get a
    # negative-infinity loss by just cranking up noise. Set high to disable.
    div_ceiling_k: float = 3.0
    score_sigma_px: float = 32.0   # width of the confidence target
    coord_weight: float = 1.0
    score_weight: float = 100.0    # deeptangle uses ~1e2
    outside_penalty: float = 0.0   # optional penalty for preds fully outside canvas


def _safe_norm(v: Array, axis: int = -1, eps: float = 1e-6) -> Array:
    """jnp.linalg.norm has inf gradient at 0 (d/dx sqrt(0) = ∞), which produces
    NaN during backprop when two curves match exactly. `sqrt(sum²+ε)` stays
    differentiable at zero."""
    return jnp.sqrt(jnp.sum(v ** 2, axis=axis) + eps)


def _pairwise_curve_dist(a: Array, b: Array) -> Array:
    """Symmetric-per-vertex L2 curve distance (accounting for reverse skeleton).

    Args:
      a: (..., K, 2)      b: (..., K, 2)
    Returns:
      (...) mean per-vertex distance between the closer of a↔b or a↔reverse(b).
    """
    d1 = _safe_norm(a - b, axis=-1).mean(axis=-1)
    d2 = _safe_norm(a - b[..., ::-1, :], axis=-1).mean(axis=-1)
    return jnp.minimum(d1, d2)


def compute_energy_loss(
    curves_a: Array, curves_b: Array,
    scores_a: Array, scores_b: Array,
    gt_curves: Array, gt_valid: Array,
    cfg_e: EnergyLossConfig,
) -> tuple[Array, dict]:
    """Combined energy-score + confidence loss.

    Shapes (single example):
      curves_{a,b}: (grid_h, grid_w, n_suggestions, K, 2)  — from 2 noise draws
      scores_{a,b}: (grid_h, grid_w, n_suggestions)
      gt_curves:    (G_max, K, 2)  — padded GT skeletons
      gt_valid:     (G_max,) bool  — mask of real GT entries
    """
    # Flatten prediction grid → (P, K, 2) with P = grid_h*grid_w*n_sug
    P = curves_a.shape[0] * curves_a.shape[1] * curves_a.shape[2]
    K = curves_a.shape[-2]
    ca = curves_a.reshape(P, K, 2)
    cb = curves_b.reshape(P, K, 2)
    sa = scores_a.reshape(P)
    sb = scores_b.reshape(P)

    # Pairwise pred-vs-gt distances: (P, G_max)
    def pair_dist(preds):
        d1 = _safe_norm(preds[:, None] - gt_curves[None], axis=-1).mean(-1)
        d2 = _safe_norm(preds[:, None] - gt_curves[None, :, ::-1, :],
                          axis=-1).mean(-1)
        return jnp.minimum(d1, d2)                                # (P, G)

    d_a = pair_dist(ca)                                            # (P, G)
    d_b = pair_dist(cb)                                            # (P, G)

    # Mask invalid GT columns (make them large so argmin ignores them)
    INF = 1e6
    valid_mask = gt_valid[None].astype(jnp.float32)
    d_a_masked = d_a * valid_mask + INF * (1.0 - valid_mask)
    d_b_masked = d_b * valid_mask + INF * (1.0 - valid_mask)

    # Nearest-GT distance per prediction — this is our accuracy term.
    nearest_a = d_a_masked.min(axis=1)                             # (P,)
    nearest_b = d_b_masked.min(axis=1)

    n_gt = jnp.maximum(gt_valid.sum().astype(jnp.float32), 1.0)

    # ---- Accuracy term: for each GT, how close is the model's best prediction?
    # This ensures the model actually TRIES to hit each GT (not just the easy ones).
    #   min over predictions of (distance from pred to GT)
    best_pred_per_gt_a = d_a_masked.min(axis=0) * valid_mask[0]     # (G,)
    best_pred_per_gt_b = d_b_masked.min(axis=0) * valid_mask[0]
    acc_a = best_pred_per_gt_a.sum() / n_gt
    acc_b = best_pred_per_gt_b.sum() / n_gt

    # ---- Diversity term: spread between the two samples on matched positions.
    # Each slot's contribution: distance between its two predictions.
    div_per_slot = _pairwise_curve_dist(ca, cb)                    # (P,)
    diversity = div_per_slot.mean()

    # ---- Score target: exp(-d²/σ²) toward NEAREST GT
    def score_target(nearest):
        return jnp.exp(-(nearest ** 2) / (cfg_e.score_sigma_px ** 2))
    s_tgt_a = jax.lax.stop_gradient(score_target(nearest_a))
    s_tgt_b = jax.lax.stop_gradient(score_target(nearest_b))
    score_loss = ((sa - s_tgt_a) ** 2 + (sb - s_tgt_b) ** 2).mean() * 0.5

    # Combine
    accuracy = 0.5 * (acc_a + acc_b)
    # Cap diversity so the model can't gain unbounded reward by emitting garbage.
    div_capped = jnp.minimum(diversity,
                              cfg_e.div_ceiling_k * jax.lax.stop_gradient(accuracy))
    energy = cfg_e.coord_weight * accuracy - cfg_e.beta * 0.5 * div_capped
    total = energy + cfg_e.score_weight * score_loss

    stats = dict(
        loss_total=total,
        loss_accuracy=accuracy,
        loss_diversity=diversity,
        loss_diversity_capped=div_capped,
        loss_score=score_loss,
        diversity_over_accuracy=diversity / jnp.maximum(accuracy, 1e-3),
        mean_score_a=sa.mean(),
        mean_score_target=s_tgt_a.mean(),
        n_gt=n_gt,
    )
    return total, stats


def gt_curves_from_sim(sim_out: dict, n_max_gt: int) -> tuple[Array, Array]:
    """Pack (curves, valid) from a sim clip for the loss.

    sim_out["curves"]: (T, N_flag, K, 2), sim_out["flagella"]["alive"]: (N_flag,)
    We supervise on the MID frame.
    """
    T = sim_out["curves"].shape[0]
    mid_curves = sim_out["curves"][T // 2]                         # (N_flag, K, 2)
    alive = sim_out["flagella"]["alive"]                           # (N_flag,)
    N = mid_curves.shape[0]
    K = mid_curves.shape[-2]
    assert N <= n_max_gt, f"n_max_flagella={N} > n_max_gt={n_max_gt}"
    # Right-pad to n_max_gt so tensor is static-shaped
    pad_curves = jnp.zeros((n_max_gt - N, K, 2), dtype=mid_curves.dtype)
    pad_valid = jnp.zeros((n_max_gt - N,), dtype=jnp.bool_)
    gt_curves = jnp.concatenate([mid_curves, pad_curves], axis=0)
    gt_valid = jnp.concatenate([alive, pad_valid], axis=0)
    return gt_curves, gt_valid
