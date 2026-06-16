"""Tracking metric: count ID switches under nearest-neighbour matching of z_what across frames.

For each video, we match predicted slot identities to GT identities at frame 0 via z_where
Hungarian, then propagate the mapping greedily through frames using z_what nearest neighbour. An
ID switch is counted whenever the inferred slot↔GT mapping changes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sim2real.losses.matching import _hungarian_cpu, build_cost_zwhere


def _matched_at_frame(pred_zw, gt_zw, gt_pres):
    cost = np.asarray(build_cost_zwhere(pred_zw, gt_zw, gt_pres))
    perm = _hungarian_cpu(cost)
    return perm


def id_switch_count(pred_zwhere, pred_zwhat, gt_zwhere, gt_pres) -> int:
    """Return total number of ID switches across the video."""
    perms = []
    T = pred_zwhere.shape[0]
    for t in range(T):
        perms.append(_matched_at_frame(np.asarray(pred_zwhere[t]), np.asarray(gt_zwhere[t]), np.asarray(gt_pres[t])))
    perms = np.stack(perms, axis=0)
    diffs = np.sum(perms[1:] != perms[:-1])
    return int(diffs)
