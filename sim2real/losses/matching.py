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


def build_cost_zwhere(pred_zwhere: Array, gt_zwhere: Array, gt_pres: Array) -> Array:
    """Cost matrix for matching predicted slots to GT slots using a single z_where frame.

    Args:
      pred_zwhere: (N, 3)
      gt_zwhere:   (N, 3)
      gt_pres:     (N,)    1 = real GT slot, 0 = padding (we set cost huge for dead GT to avoid matching).

    Returns:
      (N, N) cost matrix.
    """
    diff = pred_zwhere[:, None, :] - gt_zwhere[None, :, :]                            # (N, N, 3)
    base = jnp.sum(diff * diff, axis=-1)                                              # (N, N)
    # Heavy penalty for matching against a padding (dead) GT slot.
    penalty = (1.0 - gt_pres)[None, :] * 1e6
    return base + penalty


def hungarian_per_frame(pred_zwhere: Array, gt_zwhere: Array, gt_pres: Array) -> Array:
    """Hungarian for one frame. All inputs leading-dim N (slot count). Returns perm of shape (N,)."""
    cost = build_cost_zwhere(pred_zwhere, gt_zwhere, gt_pres)
    return hungarian(cost)


def gather_along_slots(pred_arr: Array, perm: Array) -> Array:
    """`pred_arr[perm]` along the slot axis — works for any trailing shape."""
    return pred_arr[perm]
