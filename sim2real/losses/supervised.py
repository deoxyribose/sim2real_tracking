"""Supervised losses applied after Hungarian matching:

- L_where: MSE on z_where (only for alive GT slots).
- L_pres: two-sided BCE-from-logits (numerically stable at saturation; works on the soft pres
  logit rather than the Gumbel-ST hard output).
- L_mask: BCE + soft Dice on predicted vs GT masks, **including padding slots** (whose target
  becomes all-zeros) so the seg head learns to suppress dead-slot output.

All operate on (T, N, ...) tensors already aligned by the matching perm.
"""

from __future__ import annotations

import jax
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


def bce_from_logits(
    logit: Array,
    gt_pres: Array,
    focal_gamma: float = 0.0,
    focal_alpha: float = 0.5,
) -> Array:
    """Two-sided BCE (optionally focal) computed on logits, numerically stable at saturation.

    Plain BCE (focal_gamma=0):
        L = mean( softplus(logit) - gt · logit )
    Focal BCE (focal_gamma>0, Lin et al 2017):
        L = mean( alpha_t · (1-pt)^gamma · -log(pt) )
    where pt = sigmoid(z) if gt=1 else 1-sigmoid(z), and alpha_t = alpha if gt=1 else 1-alpha.
    Numerically stable via softplus/sigmoid on ±z.

    focal_alpha in [0,1] is the positive-class weight (up-weight positives with alpha>0.5).
    """
    if focal_gamma <= 0.0 and focal_alpha == 0.5:
        return jnp.mean(jax.nn.softplus(logit) - gt_pres * logit)
    # -log(pt): softplus(-z) for pos, softplus(z) for neg.
    neg_log_pt = jnp.where(gt_pres > 0.5, jax.nn.softplus(-logit), jax.nn.softplus(logit))
    # (1 - pt)^gamma: sigmoid(-z)^g for pos, sigmoid(z)^g for neg.
    if focal_gamma > 0.0:
        mod = jnp.where(gt_pres > 0.5, jax.nn.sigmoid(-logit), jax.nn.sigmoid(logit)) ** focal_gamma
    else:
        mod = jnp.ones_like(logit)
    alpha_t = jnp.where(gt_pres > 0.5, focal_alpha, 1.0 - focal_alpha)
    return jnp.mean(alpha_t * mod * neg_log_pt)


def pres_temporal_smooth(pres_logit: Array) -> Array:
    """Penalize frame-to-frame variation of per-slot pres. Sim guarantees GT pres is constant
    across T (cells never appear/disappear), so any temporal variation is a model artifact.

    MSE in probability space: strongest gradient exactly when the slot is flipping between
    confident classes (which is the pathology we're targeting).

    pres_logit: (T, N).
    """
    pres = jax.nn.sigmoid(pres_logit)
    return jnp.mean((pres[1:] - pres[:-1]) ** 2)


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


def glimpse_mask_mse(
    mask_logit_patch: Array,   # (T, N, gh, gw, 1) — matched (already Hungarian-permuted)
    gt_masks_canvas: Array,    # (T, N, H, W)       — full-canvas GT masks
    gt_zwhere: Array,          # (T, N, 5)           — 5-dim affine
    alive: Array,              # (T, N)
) -> Array:
    """Per-slot MSE between sigmoid(decoder mask patch) and the GT mask read into glimpse space.

    Cellulose recipe (Kosiorek-derived). In glimpse coords the foreground/background pixel
    counts are ~balanced (the slot's z_where scale matches the cell size), so the loss does
    not get hijacked by the 1:500 imbalance that exists in canvas coords. This is what
    avoids the "filled rectangle" attractor of canvas-space mask supervision.

    GT masks are stn_read at the slot's z_where, giving each slot's local GT crop. Compared
    to the decoder's directly-output mask patch (sigmoid'd) — no stn_write round-trip.
    """
    from sim2real.model.stn import stn_read
    gh = mask_logit_patch.shape[-3]

    def read_one(gt_m, zw):
        # gt_m: (H, W). stn_read expects (H, W, C).
        return stn_read(gt_m[..., None], zw, gh)                                          # (gh, gh, 1)

    # vmap over (T, N).
    gt_patches = jax.vmap(jax.vmap(read_one))(gt_masks_canvas, gt_zwhere)                 # (T, N, gh, gh, 1)
    pred_patches = jax.nn.sigmoid(mask_logit_patch)
    sq = jnp.mean((pred_patches - gt_patches) ** 2, axis=(-1, -2, -3))                    # (T, N)
    return jnp.sum(sq * alive) / (jnp.sum(alive) + 1e-6)


def slot_contrast_loss(
    z_what: Array,     # (T, N, Zw) — predicted (unmatched — slot index IS identity here)
    alive: Array,      # (T, N)
    tau: float = 0.1,
) -> Array:
    """SlotContrast temporal contrastive on z_what (Manasyan et al., CVPR 2025).

    Positive pair: (z_what[t, n], z_what[t+1, n]) — same slot across consecutive frames.
    Negatives: {z_what[t+1, m] : m != n} — other slots at same target frame.
    InfoNCE at temperature tau. Only counted for slots alive at BOTH t and t+1.

    Uses the predicted z_what without Hungarian matching, because slot index IS identity by
    construction (teacher-forced z_pres + SAVi frame-0 bootstrap keep slot n aligned to GT
    slot n across the video).
    """
    z = z_what / (jnp.linalg.norm(z_what, axis=-1, keepdims=True) + 1e-6)                  # (T, N, Zw)
    z_t = z[:-1]                                                                            # (T-1, N, Zw)
    z_tp1 = z[1:]                                                                           # (T-1, N, Zw)
    sim = jnp.einsum('tnd,tmd->tnm', z_t, z_tp1) / tau                                       # (T-1, N, N)
    log_softmax = jax.nn.log_softmax(sim, axis=-1)                                           # (T-1, N, N)
    # Positive is diagonal (n == m)
    diag = jnp.diagonal(log_softmax, axis1=-2, axis2=-1)                                     # (T-1, N)
    a = alive[:-1] * alive[1:]                                                              # (T-1, N)
    return -jnp.sum(diag * a) / (jnp.sum(a) + 1e-6)


def glimpse_appear_mse(
    appear_patch: Array,       # (T, N, gh, gw, 1) — matched (Hungarian-permuted); values in [0,1]
    gt_video: Array,           # (T, H, W, C)      — same video for all slots (single-video call)
    gt_masks_canvas: Array,    # (T, N, H, W)      — foreground weighting
    gt_zwhere: Array,          # (T, N, 5)         — GT z_where
    alive: Array,              # (T, N)
) -> Array:
    """Per-slot MSE between the decoder's raw appearance glimpse and the GT image cropped at
    the same z_where, weighted by the GT mask patch (foreground only).

    This is the appearance analog of `glimpse_mask_mse`. Trains the glimpse decoder and z_what
    from a per-slot, per-glimpse target — no compositing, so the alpha-average / mean-image
    degeneracies of full-composite pixel MSE cannot apply.
    """
    from sim2real.model.stn import stn_read
    gh = appear_patch.shape[-3]

    def read_v(v, zw):     # v: (H, W, C), zw: (5,)
        return stn_read(v, zw, gh)

    def read_m(m, zw):     # m: (H, W), zw: (5,)
        return stn_read(m[..., None], zw, gh)

    def per_frame(v_t, zw_t, m_t):
        gt_appear_p = jax.vmap(lambda zw: read_v(v_t, zw))(zw_t)                          # (N, gh, gh, C)
        gt_mask_p = jax.vmap(read_m)(m_t, zw_t)                                           # (N, gh, gh, 1)
        return gt_appear_p, gt_mask_p

    gt_appear, gt_mask = jax.vmap(per_frame)(gt_video, gt_zwhere, gt_masks_canvas)        # (T, N, gh, gh, {C,1})

    err = (appear_patch - gt_appear) ** 2                                                 # (T, N, gh, gh, C)
    fg = gt_mask                                                                          # (T, N, gh, gh, 1)
    num = jnp.sum(err * fg, axis=(-1, -2, -3))                                            # (T, N)
    den = jnp.sum(fg, axis=(-1, -2, -3)) + 1e-6                                           # (T, N)
    per_slot = num / den
    return jnp.sum(per_slot * alive) / (jnp.sum(alive) + 1e-6)


def mask_loss(
    pred_mask: Array,
    gt_mask: Array,
    alive: Array,
    dice_weight: float = 1.0,
    focal_gamma: float = 0.0,
    focal_alpha: float = 0.5,
) -> Array:
    """BCE (optionally focal) over ALL slots + soft-Dice over ALIVE slots only.

    When `focal_gamma == 0`, the BCE term is the standard cross-entropy. When `focal_gamma > 0`,
    each per-pixel loss is multiplied by `(1−p)^γ` for positive pixels and `p^γ` for negatives,
    downweighting easy examples (Lin et al. 2017, "Focal Loss for Dense Object Detection").
    `focal_alpha` is the positive-class weight in [0, 1].

    Rationale for the split:
    - BCE on ALL slots: pulls pred → 0 on padding slots (no "red rectangle" failure).
    - Dice on ALIVE only: avoids the degenerate `1 − 2·0/(p+0+ε) ≈ 1` constant floor on dead slots.
    - Focal: counters the "predict 0 everywhere" coasting trap on highly imbalanced masks where
      most pixels are 0 — gradient on hard (cell-boundary) pixels stays loud as the model improves.

    Args:
      pred_mask: (T, N, H, W) in [0,1].
      gt_mask:   (T, N, H, W) in [0,1].
      alive:     (T, N) in {0, 1} — GT z_pres.
    """
    eps = 1e-6
    gt_effective = gt_mask * alive[..., None, None]                                   # zero out dead
    p = jnp.clip(pred_mask, eps, 1.0 - eps)

    if focal_gamma > 0:
        pos = focal_alpha * (1.0 - p) ** focal_gamma * jnp.log(p)
        neg = (1.0 - focal_alpha) * (p ** focal_gamma) * jnp.log(1.0 - p)
        bce = -(gt_effective * pos + (1.0 - gt_effective) * neg)
    else:
        bce = -(gt_effective * jnp.log(p) + (1.0 - gt_effective) * jnp.log(1.0 - p))   # (T,N,H,W)
    bce_per_slot = jnp.mean(bce, axis=(-1, -2))                                        # (T, N)

    inter = jnp.sum(p * gt_effective, axis=(-1, -2))
    denom = jnp.sum(p + gt_effective, axis=(-1, -2)) + eps
    dice_per_slot = 1.0 - 2.0 * inter / denom                                          # (T, N)

    L_bce = jnp.mean(bce_per_slot)
    L_dice_num = jnp.sum(dice_per_slot * alive)
    L_dice_den = jnp.sum(alive) + 1e-6
    L_dice = L_dice_num / L_dice_den
    return L_bce + dice_weight * L_dice
