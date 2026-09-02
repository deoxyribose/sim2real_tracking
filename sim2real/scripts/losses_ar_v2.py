"""Alternative attachment losses for AR model — for the diagnostic test.

Variants:
  score_mode:
    - "wide-gauss"  (baseline v8): BCE(sigmoid, exp(-d^2/sigma^2)), sigma=12
    - "hard-mask"   : BCE, but only inside a mask (cell within R of any GT
                      attachment). Removes the millions of near-zero targets.
    - "focal"       : focal BCE (γ=2) on the wide-gauss target, so positives
                      dominate the gradient.
  coord_mode:
    - "min-over-all" (baseline v8): per-GT, min-over-all preds of L2 distance.
                     Only 1 pred per GT gets gradient.
    - "per-cell-in-radius": every (cell, sugg) whose center is within R of any
                     GT attachment learns to regress to the NEAREST GT.
                     Dense gradient.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from sim2real.model.unet_ar import (
    UNetARConfig, encode_gt_polar_steps, encode_gt_with_stop,
    rotated_patch_batched, sample_batched_noise, unpack_attachment,
)


def _grid_cell_centers(cfg: UNetARConfig) -> jnp.ndarray:
    ys = jnp.arange(cfg.grid_h) * cfg.grid_stride + cfg.grid_stride / 2
    xs = jnp.arange(cfg.grid_w) * cfg.grid_stride + cfg.grid_stride / 2
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    return jnp.stack([yy, xx], axis=-1).astype(jnp.float32)


def make_loss_fn(score_mode: str = "wide-gauss",
                   coord_mode: str = "min-over-all",
                   score_sigma_px: float = 12.0,
                   mask_radius_px: float = 16.0,
                   focal_gamma: float = 2.0,
                   knot_label_smoothing: float = 0.0,
                   scheduled_sampling: bool = False,
                   use_stop_head: bool = False,
                   stop_weight: float = 1.0,
                   smoothness_weight: float = 0.0):

    def loss_fn(params, batch, key, backbone, attach_head, knot_gen,
                cfg: UNetARConfig, coord_weight: float = 5.0,
                score_weight: float = 5.0, knot_weight: float = 1.0,
                p_teacher_force: float = 1.0):
        video, static_med, gt_skels, gt_valid = batch
        B = video.shape[0]

        key_n, key_r = jax.random.split(key)
        noise = sample_batched_noise(key_n, B, cfg)
        full_res, grid = backbone.apply(params["backbone"], video, noise, static_med,
                                          train=True)
        attach = attach_head.apply(params["attach"], grid)         # (B, gh, gw, n_sug, 3)
        f = unpack_attachment(attach)
        cell_centers = _grid_cell_centers(cfg)                     # (gh, gw, 2)
        pred_pos = cell_centers[None, :, :, None, :] + jnp.stack([f["dy"], f["dx"]], -1)

        def per_ex_attach(pos, scores, gt, valid):
            # pos: (gh, gw, n_sug, 2)   gt: (N_gt, K+1, 2)   valid: (N_gt,)
            gt_att = gt[:, 0]                                     # (N_gt, 2)
            valid_mask = valid.astype(jnp.float32)[:, None, None, None]

            # ---- SCORE ----------------------------------------------------
            d_pos = jnp.linalg.norm(gt_att[:, None, None, None, :] - pos[None], axis=-1)
            d_pos_masked = d_pos * valid_mask + 1e6 * (1 - valid_mask)
            d_pos_min = d_pos_masked.min(axis=0)                  # (gh, gw, n_sug)
            s_target = jnp.exp(-(d_pos_min ** 2) / (score_sigma_px ** 2))
            s_target = jax.lax.stop_gradient(s_target)
            eps = 1e-6
            s_pred = jax.nn.sigmoid(scores)
            per_slot_bce = -(s_target * jnp.log(s_pred + eps)
                              + (1 - s_target) * jnp.log(1 - s_pred + eps))

            if score_mode == "wide-gauss":
                score_loss = per_slot_bce.mean()
            elif score_mode == "hard-mask":
                # Only supervise (cell, sugg) slots whose grid-center is within
                # mask_radius of ANY GT. This throws out the sea of near-zero targets.
                gc = cell_centers[None]                            # (1, gh, gw, 2)
                d_cell = jnp.linalg.norm(
                    gt_att[:, None, None, :] - gc, axis=-1)        # (Ngt, gh, gw)
                v = valid.astype(jnp.float32)[:, None, None]     # (Ngt, 1, 1)
                d_cell_m = d_cell * v + 1e6 * (1 - v)
                d_cell_min = d_cell_m.min(0)                       # (gh, gw)
                mask = (d_cell_min <= mask_radius_px).astype(jnp.float32)
                # broadcast to n_sug
                mask_bcst = mask[:, :, None]
                score_loss = (per_slot_bce * mask_bcst).sum() / (mask_bcst.sum() + 1e-6)
            elif score_mode == "focal":
                # focal BCE: (1 - p_t)^gamma * bce_t
                p_t = s_pred * s_target + (1 - s_pred) * (1 - s_target)
                focal_w = jnp.power(1.0 - p_t, focal_gamma)
                score_loss = (focal_w * per_slot_bce).mean()
            else:
                raise ValueError(f"unknown score_mode: {score_mode}")

            # ---- COORD ----------------------------------------------------
            if coord_mode == "min-over-all":
                best_per_gt = d_pos.min(axis=(1, 2, 3)) * valid_mask[..., 0, 0, 0]
                n_gt = jnp.maximum(valid.sum().astype(jnp.float32), 1)
                coord_loss = best_per_gt.sum() / n_gt
            elif coord_mode == "per-cell-in-radius":
                # For each slot, if grid-center is within R of GT, regress to
                # the NEAREST GT attachment.
                gc = cell_centers[None]                            # (1, gh, gw, 2)
                d_cell = jnp.linalg.norm(
                    gt_att[:, None, None, :] - gc, axis=-1)        # (Ngt, gh, gw)
                v = valid.astype(jnp.float32)[:, None, None]     # (Ngt, 1, 1)
                d_cell_m = d_cell * v + 1e6 * (1 - v)
                nearest_gt_idx = jnp.argmin(d_cell_m, axis=0)      # (gh, gw)
                nearest_gt = gt_att[nearest_gt_idx]                # (gh, gw, 2)
                # slot regresses to nearest GT
                offsets = nearest_gt[:, :, None, :] - pos          # (gh, gw, n_sug, 2)
                slot_l1 = jnp.abs(offsets).sum(-1)                 # (gh, gw, n_sug)
                d_cell_min = d_cell_m.min(0)                       # (gh, gw)
                mask = (d_cell_min <= mask_radius_px).astype(jnp.float32)[:, :, None]
                coord_loss = (slot_l1 * mask).sum() / (mask.sum() + 1e-6)
            else:
                raise ValueError(f"unknown coord_mode: {coord_mode}")

            return coord_loss, score_loss

        coord_l, score_l = jax.vmap(per_ex_attach)(
            pred_pos, f["score"], gt_skels, gt_valid)
        coord_l = coord_l.mean(); score_l = score_l.mean()

        # ---- Knot loss: teacher-forced + optional scheduled sampling ------
        def smoothed_ce_row(logits, target_bin, n_bins):
            logp = jax.nn.log_softmax(logits, -1)
            hard = -logp[target_bin]
            if knot_label_smoothing > 0.0:
                uni = -logp.mean()
                return (1 - knot_label_smoothing) * hard \
                     + knot_label_smoothing * uni
            return hard

        def per_ex_knot(fmap, gt_skel_batch, valid, ex_key):
            def per_gt(skel, is_valid, gkey):
                if use_stop_head:
                    (att, tangents, d_ang_bin, d_step_bin,
                       valid_mask, stop_target) = encode_gt_with_stop(skel, cfg)
                else:
                    att, tangents, d_ang_bin, d_step_bin = encode_gt_polar_steps(
                        skel, cfg)
                    valid_mask = jnp.ones(cfg.n_knots, dtype=jnp.bool_)
                    stop_target = jnp.zeros(cfg.n_knots, dtype=jnp.float32)
                K = tangents.shape[0]
                # Build skel from attachment + tangents for centers (matches
                # what encode_gt_polar_steps used). Re-run resample_polyline
                # via arc positions used inside encode:
                #   centers = attachment + cumulative offsets... too messy;
                # simplest: use the tangents to reconstruct positions.
                # Alternative: recompute centers via prev-position + step
                # from the GT bins. But that costs precision. We keep
                # centers pointing at the ORIGINAL skel positions by simply
                # regenerating them from the resampled arc positions used
                # inside encode_gt_polar_steps → we duplicate that logic here.
                # For simplicity we compute centers by cumulative reconstruction
                # of ideal positions (used only as patch centers).
                d_ang_c = cfg.angle_bin_centers[d_ang_bin]
                d_step_c = cfg.step_bin_centers[d_step_bin]
                def _pos_scan(carry, i):
                    pos, tan = carry
                    new_tan = tan + d_ang_c[i]
                    step = d_step_c[i]
                    new_pos = jnp.stack([pos[0] + step * jnp.sin(new_tan),
                                          pos[1] + step * jnp.cos(new_tan)])
                    return (new_pos, new_tan), pos
                (_, _), centers = jax.lax.scan(
                    _pos_scan, (att, jnp.array(0.0)), jnp.arange(K))
                gt_centers = centers                              # (K, 2)
                gt_prev_tan = jnp.concatenate([jnp.zeros(1), tangents[:-1]])

                if not scheduled_sampling:
                    # Pure teacher-forcing — batched (fast) path
                    patches = rotated_patch_batched(
                        fmap, gt_centers, gt_prev_tan, cfg.patch_size)
                    kg_out = knot_gen.apply(params["knot"], patches)
                    if use_stop_head:
                        angle_logits, step_logits, stop_logits = kg_out
                    else:
                        angle_logits, step_logits = kg_out
                        stop_logits = jnp.zeros(K)
                    a_loss = jax.vmap(smoothed_ce_row, in_axes=(0, 0, None))(
                        angle_logits, d_ang_bin, cfg.n_angle_bins)
                    s_loss = jax.vmap(smoothed_ce_row, in_axes=(0, 0, None))(
                        step_logits, d_step_bin, cfg.n_step_bins)
                else:
                    # Scheduled sampling — per-knot coin flip; must scan
                    def step_fn(carry, k):
                        pos, tangent, kk = carry
                        # per-knot coin: 1 = teacher-force (use GT), 0 = self-context
                        k1, k2, k3 = jax.random.split(
                            jax.random.fold_in(gkey, kk), 3)
                        coin = jax.random.uniform(k1) < p_teacher_force
                        # First knot is always teacher-forced (it's the attachment)
                        force_gt = jnp.logical_or(coin, kk == 0)
                        center = jnp.where(force_gt, gt_centers[k], pos)
                        prev_t = jnp.where(force_gt, gt_prev_tan[k], tangent)
                        patch = rotated_patch_batched(
                            fmap, center[None], prev_t[None], cfg.patch_size)
                        a_lg, s_lg = knot_gen.apply(params["knot"], patch)
                        # Loss: always vs GT bin at step k
                        al = smoothed_ce_row(a_lg[0], d_ang_bin[k], cfg.n_angle_bins)
                        sl = smoothed_ce_row(s_lg[0], d_step_bin[k], cfg.n_step_bins)
                        # Sample from model's categorical to propagate self-state
                        a_bin_s = jax.random.categorical(k2, a_lg[0])
                        s_bin_s = jax.random.categorical(k3, s_lg[0])
                        d_ang_s = cfg.angle_bin_centers[a_bin_s]
                        d_step_s = cfg.step_bin_centers[s_bin_s]
                        new_tan = prev_t + d_ang_s
                        new_pos = jnp.stack([center[0] + d_step_s * jnp.sin(new_tan),
                                              center[1] + d_step_s * jnp.cos(new_tan)])
                        return (new_pos, new_tan, kk + 1), (al, sl)
                    (_, _, _), (a_loss, s_loss) = jax.lax.scan(
                        step_fn, (skel[0], jnp.array(0.0), jnp.int32(0)),
                        jnp.arange(K))
                    stop_logits = jnp.zeros(K)      # SS + stop not supported yet

                # Mask by valid: only valid knots count in knot CE
                vmask = valid_mask.astype(jnp.float32)
                a_loss = (a_loss * vmask).sum() / jnp.maximum(vmask.sum(), 1)
                s_loss = (s_loss * vmask).sum() / jnp.maximum(vmask.sum(), 1)

                # Stop BCE (all knots contribute) if stop head is on
                if use_stop_head:
                    logp1 = jax.nn.log_sigmoid(stop_logits)
                    logp0 = jax.nn.log_sigmoid(-stop_logits)
                    stop_bce = -(stop_target * logp1
                                  + (1.0 - stop_target) * logp0).mean()
                else:
                    stop_bce = jnp.array(0.0)

                # Smoothness prior: penalize (E[Δang_k] - E[Δang_{k-1}])²
                if smoothness_weight > 0.0:
                    # E[Δang_k] under the categorical
                    exp_ang = (jax.nn.softmax(angle_logits, -1)
                                * cfg.angle_bin_centers[None]).sum(-1)  # (K,)
                    d_exp = exp_ang[1:] - exp_ang[:-1]           # (K-1,)
                    # Only count transitions where BOTH knots are valid
                    v_pair = (vmask[1:] * vmask[:-1])            # (K-1,)
                    sm = (d_exp ** 2 * v_pair).sum() / jnp.maximum(v_pair.sum(), 1)
                else:
                    sm = jnp.array(0.0)

                per_gt_loss = (a_loss + s_loss
                                + stop_weight * stop_bce
                                + smoothness_weight * sm)
                return per_gt_loss * is_valid.astype(jnp.float32), stop_bce, sm
            gt_keys = jax.random.split(ex_key, gt_skel_batch.shape[0])
            losses, stop_bces, sms = jax.vmap(per_gt)(
                gt_skel_batch, valid, gt_keys)
            n_gt = jnp.maximum(valid.sum().astype(jnp.float32), 1)
            return losses.sum() / n_gt, stop_bces.mean(), sms.mean()

        ex_keys = jax.random.split(key_r, video.shape[0])
        knot_l_all, stop_l_all, sm_l_all = jax.vmap(per_ex_knot)(
            full_res, gt_skels, gt_valid, ex_keys)
        knot_l = knot_l_all.mean()
        stop_l = stop_l_all.mean()
        sm_l = sm_l_all.mean()

        total = (coord_weight * coord_l + score_weight * score_l
                  + knot_weight * knot_l)
        stats = dict(loss_total=total, loss_coord=coord_l,
                      loss_score=score_l, loss_knot=knot_l,
                      loss_stop=stop_l, loss_smooth=sm_l)
        return total, stats
    return loss_fn
