"""Supervised losses applied after Hungarian matching:

- L_where: MSE on z_where (only for alive GT slots).
- L_pres: two-sided BCE-from-logits (numerically stable at saturation; works on the soft pres
  logit rather than the Gumbel-ST hard output).
- L_mask: BCE + soft Dice on predicted vs GT masks, **including padding slots** (whose target
  becomes all-zeros) so the seg head learns to suppress dead-slot output.

All operate on (T, N, ...) tensors already aligned by the matching perm.
"""

from __future__ import annotations

import jax.nn
import jax.numpy as jnp

Array = jnp.ndarray


def masked_mse(pred: Array, gt: Array, alive: Array) -> Array:
    """`alive` is (T, N) — 1 for valid GT slots. Squared error averaged over the trailing dims and
    summed over alive entries, then normalized by total alive count."""
    diff_sq = jnp.sum((pred - gt) ** 2, axis=-1)                                     # (T, N)
    num = jnp.sum(diff_sq * alive)
    den = jnp.sum(alive) + 1e-6
    return num / den


def bce_from_logits(logit: Array, gt_pres: Array) -> Array:
    """Two-sided BCE computed directly on logits — numerically stable at saturation.

    L = mean( logsumexp(0, -z) + (1 - gt) · z ) i.e. mean over (T, N) of:
        log(1 + exp(z)) - gt · z
    """
    return jnp.mean(jax.nn.softplus(logit) - gt_pres * logit)


def bce_one_sided(pred_pres: Array, gt_pres: Array, eps: float = 1e-6) -> Array:
    """Deprecated — kept for the adapt loop. Use bce_from_logits in pretrain."""
    pred = jnp.clip(pred_pres, eps, 1.0 - eps)
    return -jnp.mean((1.0 - gt_pres) * jnp.log(1.0 - pred))


def bce_two_sided(pred_pres: Array, gt_pres: Array, eps: float = 1e-6) -> Array:
    pred = jnp.clip(pred_pres, eps, 1.0 - eps)
    return -jnp.mean(gt_pres * jnp.log(pred) + (1.0 - gt_pres) * jnp.log(1.0 - pred))


def group_supervision_nll(g_post_matched: Array, alive: Array) -> Array:
    """Supervised cross-entropy: per Hungarian-matched slot n, target group = n.

    Interpretation: with K ≥ N_max, this forces each slot to use a unique group id, i.e.
    "each instance is its own group". At test time the model is expected to keep doing this
    by inertia even when the supervision is removed.

    Args:
      g_post_matched: (T, N, K) — alignment via the matching permutation.
      alive:          (T, N)    — GT z_pres mask.

    Returns: scalar averaged over alive (T, N).
    """
    T, N, K = g_post_matched.shape
    # gather q(g_n = n) along the slot axis
    slot_idx = jnp.arange(N)
    # g_post_matched[:, n, n] for each n
    target_probs = g_post_matched[:, slot_idx, slot_idx]                              # (T, N)
    nll = -jnp.log(jnp.clip(target_probs, 1e-8, 1.0))
    return jnp.sum(nll * alive) / (jnp.sum(alive) + 1e-6)


def group_temporal_kl(g_post_matched: Array, alive: Array) -> Array:
    """Symmetric KL between consecutive frames' per-slot group posterior. Pulls each slot's
    group_id to be stable across time (temporal coherence), independent of any supervision —
    so this term remains useful at unsupervised adapt time.
    """
    eps = 1e-8
    q_t = jnp.clip(g_post_matched[1:], eps, 1.0)                                       # (T-1, N, K)
    q_tm1 = jnp.clip(g_post_matched[:-1], eps, 1.0)
    kl_forward = jnp.sum(q_t * (jnp.log(q_t) - jnp.log(q_tm1)), axis=-1)               # (T-1, N)
    kl_backward = jnp.sum(q_tm1 * (jnp.log(q_tm1) - jnp.log(q_t)), axis=-1)
    sym_kl = 0.5 * (kl_forward + kl_backward)
    a = jnp.minimum(alive[1:], alive[:-1])                                              # only when both frames alive
    return jnp.sum(sym_kl * a) / (jnp.sum(a) + 1e-6)


def mask_loss(pred_mask: Array, gt_mask: Array, alive: Array, dice_weight: float = 1.0) -> Array:
    """BCE over ALL slots + soft-Dice over ALIVE slots only.

    Rationale:
    - BCE is well-defined for empty targets: it pulls pred → 0 on padding slots, which is what
      we want (no "red square over the canvas" failure).
    - Dice on empty targets is degenerate — `1 − 2·0/(p+0+ε) ≈ 1` for any small p, adding a
      constant ~`n_dead/n_total` floor to the loss with no useful gradient. So we only Dice on
      alive slots.

    Args:
      pred_mask: (T, N, H, W) in [0,1].
      gt_mask:   (T, N, H, W) in [0,1].
      alive:     (T, N) in {0, 1} — GT z_pres.
    """
    eps = 1e-6
    gt_effective = gt_mask * alive[..., None, None]                                   # zero out dead
    p = jnp.clip(pred_mask, eps, 1.0 - eps)
    bce = -(gt_effective * jnp.log(p) + (1.0 - gt_effective) * jnp.log(1.0 - p))      # (T,N,H,W)
    bce_per_slot = jnp.mean(bce, axis=(-1, -2))                                       # (T, N)

    inter = jnp.sum(p * gt_effective, axis=(-1, -2))
    denom = jnp.sum(p + gt_effective, axis=(-1, -2)) + eps
    dice_per_slot = 1.0 - 2.0 * inter / denom                                         # (T, N)

    L_bce = jnp.mean(bce_per_slot)                                                    # over ALL slots
    L_dice_num = jnp.sum(dice_per_slot * alive)
    L_dice_den = jnp.sum(alive) + 1e-6
    L_dice = L_dice_num / L_dice_den                                                  # alive only
    return L_bce + dice_weight * L_dice
