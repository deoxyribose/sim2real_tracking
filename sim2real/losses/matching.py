"""Hungarian matching between predicted slots and ground-truth slots.

Runs on host via `scipy.optimize.linear_sum_assignment`, returns int permutation indices wrapped
in `jax.lax.stop_gradient` so gradients flow only through the matched values, not the assignment.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

Array = jnp.ndarray


def _hungarian_cpu(cost: np.ndarray) -> np.ndarray:
    """Run Hungarian on a (N, N) cost matrix. Returns `perm` of length N such that `pred[perm]`
    aligns to GT order — i.e. `pred[perm[j]]` is the prediction matched to `gt[j]`. This makes
    `_apply_perm(arr, perm) = arr[perm]` produce an array indexed in GT slot order, which is what
    the supervised loss expects.
    """
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.zeros(cost.shape[0], dtype=np.int32)
    perm[col_ind] = row_ind
    return perm


def hungarian(cost: Array) -> Array:
    """Stop gradients on `cost` and call the host-side solver via `pure_callback`.

    We stop the gradient *into* the cost matrix because the Hungarian solver is not
    differentiable; gradients on the cost would be meaningless. The permutation output is also
    treated as a constant.
    """
    cost = jax.lax.stop_gradient(cost)
    perm = jax.pure_callback(
        _hungarian_cpu,
        jax.ShapeDtypeStruct(cost.shape[:1], jnp.int32),
        cost,
        vmap_method="sequential",
    )
    return perm


def build_cost_zwhere(
    pred_zwhere: Array,
    gt_zwhere: Array,
    gt_pres: Array,
    pred_pres: Array | None = None,
    pres_weight: float = 0.0,
    hard_pres_gate: bool = False,
) -> Array:
    """Cost matrix for matching predicted slots to GT slots using POSITION ONLY (tx, ty).

    We deliberately ignore (sx, sy, θ) for matching cost — those are shape descriptors and
    coupling them into identity makes the model reluctant to assign a slot whose shape isn't
    quite right yet. Position alone (z_where indices 3, 4) is what tells us "this slot is at
    this cell".

    Optionally adds a DETR-style class term `-pres_weight * p̂_pres(i)` for alive GT columns,
    biasing the matcher to send confidently-alive predictions to alive GT slots.

    When `hard_pres_gate=True`, adds a huge penalty `(1-pred_pres)*gt_pres*1e6` — dead-pred
    slots can no longer win alive-GT columns. Requires `pred_pres`. Together with the existing
    `(1-gt_pres)*1e6` on dead-GT columns, this forces alive↔alive and dead↔dead pairings.

    Args:
      pred_zwhere: (N, 5) — last 2 dims are (tx_raw, ty_raw).
      gt_zwhere:   (N, 5)
      gt_pres:     (N,)    1 = real GT slot, 0 = padding.
      pred_pres:   (N,) optional predicted pres probabilities. Ignored when pres_weight == 0
                   AND hard_pres_gate == False.
      pres_weight: scalar weight on the -p̂ term (DETR uses this instead of log-probs
                   because it's commensurable with box cost).
      hard_pres_gate: forbid dead-pred → alive-GT pairings via 1e6 penalty.

    Returns:
      (N, N) cost matrix.
    """
    pos_pred = pred_zwhere[:, -2:]                                                     # (N, 2)
    pos_gt = gt_zwhere[:, -2:]
    diff = pos_pred[:, None, :] - pos_gt[None, :, :]                                   # (N, N, 2)
    base = jnp.sum(diff * diff, axis=-1)
    if pred_pres is not None and pres_weight > 0.0:
        # Only apply the pres bonus on alive GT columns (dead columns already have the 1e6 penalty).
        pres_cost = -pres_weight * pred_pres[:, None] * gt_pres[None, :]                # (N, N)
        base = base + pres_cost
    if hard_pres_gate:
        if pred_pres is None:
            raise ValueError("hard_pres_gate requires pred_pres")
        dead_pred_penalty = (1.0 - pred_pres)[:, None] * gt_pres[None, :] * 1e6
        base = base + dead_pred_penalty
    penalty = (1.0 - gt_pres)[None, :] * 1e6
    return base + penalty


def hungarian_per_frame(
    pred_zwhere: Array,
    gt_zwhere: Array,
    gt_pres: Array,
    pred_pres: Array | None = None,
    pres_weight: float = 0.0,
    hard_pres_gate: bool = False,
) -> Array:
    """Hungarian for one frame. All inputs leading-dim N (slot count). Returns perm of shape (N,)."""
    cost = build_cost_zwhere(pred_zwhere, gt_zwhere, gt_pres, pred_pres, pres_weight,
                             hard_pres_gate=hard_pres_gate)
    return hungarian(cost)


def gather_along_slots(pred_arr: Array, perm: Array) -> Array:
    """`pred_arr[perm]` along the slot axis — works for any trailing shape."""
    return pred_arr[perm]
