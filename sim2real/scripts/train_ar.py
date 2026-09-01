"""Trainer for the autoregressive knot generator.

Loss:
  L_attach : per grid cell, BCE(score) toward exp(-d²/σ²) with d = distance
             to nearest GT attachment; L1(dy, dx) at cells within σ_px of a
             GT attachment.
  L_knot   : teacher-forced categorical cross-entropy per GT knot
             (Δangle categorical + step_len categorical). Patches are cropped
             from the encoder feature map at each GT knot position, rotated
             by the previous GT tangent so the +x axis points in the flow
             direction.
"""
from __future__ import annotations

import argparse
import pickle
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from sim2real.model.unet_ar import (
    AttachmentHead, KnotGenerator, UNetARBackbone, UNetARConfig,
    encode_gt_polar_steps, rotated_patch_batched, sample_batched_noise,
    unpack_attachment,
)
from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip


class TrainState:
    def __init__(self, params, opt_state, step):
        self.params, self.opt_state, self.step = params, opt_state, step
    def tree_flatten(self): return (self.params, self.opt_state, self.step), None
    @classmethod
    def tree_unflatten(cls, aux, ch): return cls(*ch)


jax.tree_util.register_pytree_node_class(TrainState)


def _grid_cell_centers(cfg: UNetARConfig) -> jnp.ndarray:
    ys = jnp.arange(cfg.grid_h) * cfg.grid_stride + cfg.grid_stride / 2
    xs = jnp.arange(cfg.grid_w) * cfg.grid_stride + cfg.grid_stride / 2
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    return jnp.stack([yy, xx], axis=-1).astype(jnp.float32)   # (gh, gw, 2)


def loss_fn(params, batch, key, backbone, attach_head, knot_gen,
            cfg: UNetARConfig, score_sigma_px: float = 12.0,
            coord_weight: float = 5.0, score_weight: float = 5.0,
            knot_weight: float = 1.0):
    video, static_med, gt_skels, gt_valid = batch
    B = video.shape[0]

    key_n, key_r = jax.random.split(key)
    noise = sample_batched_noise(key_n, B, cfg)
    full_res, grid = backbone.apply(params["backbone"], video, noise, static_med,
                                      train=True)
    attach = attach_head.apply(params["attach"], grid)         # (B, gh, gw, n_sug, 3)
    f = unpack_attachment(attach)

    # ---- Attachment loss ------------------------------------------------
    # Predicted attachment positions per (grid, sugg): center + (dy, dx)
    cell_centers = _grid_cell_centers(cfg)                     # (gh, gw, 2)
    pred_pos = cell_centers[None, :, :, None, :] + jnp.stack([f["dy"], f["dx"]], -1)
    #                        └ (B, gh, gw, n_sug, 2)

    def per_ex_attach(pos, scores, gt, valid):
        # pos: (gh, gw, n_sug, 2)   gt: (N_gt, K+1, 2)   valid: (N_gt,)
        # For each GT attachment (skeleton[0]), find distance to every pred
        gt_att = gt[:, 0]                                     # (N_gt, 2)
        # distances (Ngt, gh, gw, n_sug)
        d = jnp.linalg.norm(gt_att[:, None, None, None, :]
                             - pos[None], axis=-1)
        valid_mask = valid.astype(jnp.float32)[:, None, None, None]
        # For each grid cell → distance to nearest VALID gt
        d_masked = d * valid_mask + 1e6 * (1 - valid_mask)
        d_min_per_cell = d_masked.min(axis=0)                 # (gh, gw, n_sug)
        s_target = jnp.exp(-(d_min_per_cell ** 2) / (score_sigma_px ** 2))
        s_target = jax.lax.stop_gradient(s_target)
        eps = 1e-6
        s_pred = jax.nn.sigmoid(scores)
        score_loss = -(s_target * jnp.log(s_pred + eps)
                        + (1 - s_target) * jnp.log(1 - s_pred + eps)).mean()

        # Coord loss: near-attachment cells regress to their nearest GT.
        # Use mean over GT of min-distance-per-GT (like energy_loss)
        best_per_gt = d.min(axis=(1, 2, 3)) * valid_mask[..., 0, 0, 0]
        n_gt = jnp.maximum(valid.sum().astype(jnp.float32), 1)
        coord_loss = best_per_gt.sum() / n_gt

        return coord_loss, score_loss, s_target

    coord_l, score_l, _ = jax.vmap(per_ex_attach)(
        pred_pos, f["score"], gt_skels, gt_valid)
    coord_l = coord_l.mean(); score_l = score_l.mean()

    # ---- Knot loss: teacher-forced -------------------------------------
    # For each GT skeleton, encode into polar steps and get target categorical
    # bins per step.
    def per_ex_knot(fmap, gt_skel_batch, valid):
        # fmap: (H, W, C)  gt_skel_batch: (N_gt, K+1, 2)  valid: (N_gt,)
        def per_gt(skel, is_valid):
            att, tangents, d_ang_bin, d_step_bin = encode_gt_polar_steps(skel, cfg)
            # For each step k, patch is at knot k (0-indexed after attachment).
            # But we teacher-force: patch is at GT knot POSITION k (which is
            # skel[k], the START of step k+1). Wait — let's redefine:
            #   step k advances from knot k (skel[k]) to knot k+1 (skel[k+1]).
            #   The patch used to predict step k is centered on skel[k],
            #   rotated by the tangent OF STEP k-1 (i.e. tangents[k-1] or 0 for k=0).
            K = tangents.shape[0]
            centers = skel[:K]                            # (K, 2) — knot at start of each step
            prev_tangents = jnp.concatenate([jnp.zeros(1), tangents[:-1]])
            patches = rotated_patch_batched(
                fmap, centers, prev_tangents, cfg.patch_size)  # (K, P, P, C)
            angle_logits, step_logits = knot_gen.apply(
                params["knot"], patches)                   # (K, n_angle), (K, n_step)
            angle_loss = jnp.take_along_axis(
                -jax.nn.log_softmax(angle_logits, -1),
                d_ang_bin[:, None], axis=-1)[:, 0]
            step_loss = jnp.take_along_axis(
                -jax.nn.log_softmax(step_logits, -1),
                d_step_bin[:, None], axis=-1)[:, 0]
            return (angle_loss.mean() + step_loss.mean()) * is_valid.astype(jnp.float32)
        losses = jax.vmap(per_gt)(gt_skel_batch, valid)   # (N_gt,)
        n_gt = jnp.maximum(valid.sum().astype(jnp.float32), 1)
        return losses.sum() / n_gt

    knot_l = jax.vmap(per_ex_knot)(full_res, gt_skels, gt_valid).mean()

    total = (coord_weight * coord_l
              + score_weight * score_l
              + knot_weight * knot_l)
    stats = dict(loss_total=total, loss_coord=coord_l,
                  loss_score=score_l, loss_knot=knot_l)
    return total, stats


def sample_batch(key, sim_cfg: DiverseSimConfig, batch_size: int):
    keys = jax.random.split(key, batch_size)
    outs = jax.vmap(lambda k: sample_clip(k, sim_cfg))(keys)
    clip = outs["clip_median"]
    static_med = outs["temporal_median"][..., None]
    # GT skeleton: (B, N_max_flag, K+1, 2)  (curves at mid frame)
    T = clip.shape[1]
    curves = outs["curves"][:, T // 2]          # (B, N_flag, K_arc, 2)
    # Note: sim's curves have K_arc points; we treat point 0 as attachment.
    # That gives K_arc-1 "steps" per skeleton, so `n_knots` config must be
    # K_arc - 1 or we need to resample. Simplest: pass raw and let loss slice.
    valid = outs["flagella"]["alive"]
    return clip, static_med, curves, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=3e-4)
    ap.add_argument("--coord-weight", type=float, default=5.0)
    ap.add_argument("--score-weight", type=float, default=5.0)
    ap.add_argument("--knot-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--base-channels", type=int, default=48)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cfg = UNetARConfig(T=args.T, H=args.H, W=args.H,
                        base_channels=args.base_channels)
    sim_cfg = DiverseSimConfig(T=args.T, H=args.H, W=args.H)
    print(f"cfg: H={cfg.H} T={cfg.T} grid={cfg.grid_h}×{cfg.grid_w} "
          f"patch={cfg.patch_size} n_bins=(ang{cfg.n_angle_bins}, "
          f"step{cfg.n_step_bins})   n_max_flag={sim_cfg.n_max_flagella}")

    backbone = UNetARBackbone(cfg=cfg)
    attach_head = AttachmentHead(cfg=cfg)
    knot_gen = KnotGenerator(cfg=cfg)

    key = jax.random.key(args.seed)
    key, k_b, k_a, k_k = jax.random.split(key, 4)
    dummy_video = jnp.zeros((1, cfg.T, cfg.H, cfg.W))
    dummy_noise = jnp.zeros((1, cfg.H, cfg.W, 1))
    dummy_med = jnp.zeros((1, cfg.H, cfg.W, 1))
    p_backbone = backbone.init(k_b, dummy_video, dummy_noise, dummy_med, train=True)
    dummy_fr, dummy_gr = backbone.apply(p_backbone, dummy_video, dummy_noise,
                                          dummy_med, train=True)
    p_attach = attach_head.init(k_a, dummy_gr)
    p_knot = knot_gen.init(k_k, jnp.zeros((1, cfg.patch_size, cfg.patch_size,
                                              cfg.feature_channels)))
    params = dict(backbone=p_backbone, attach=p_attach, knot=p_knot)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"model params: {n_params/1e6:.2f} M")

    lr_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr,
        warmup_steps=args.warmup_steps, decay_steps=args.n_steps,
        end_value=args.lr_min)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_fn, weight_decay=args.weight_decay),
    )
    opt_state = optimizer.init(params)
    state = TrainState(params=params, opt_state=opt_state, step=jnp.array(0))

    loss_partial = partial(loss_fn, backbone=backbone, attach_head=attach_head,
                            knot_gen=knot_gen, cfg=cfg,
                            coord_weight=args.coord_weight,
                            score_weight=args.score_weight,
                            knot_weight=args.knot_weight)

    @partial(jax.jit, donate_argnums=(0,))
    def train_step(state, batch, key):
        (loss, stats), grads = jax.value_and_grad(loss_partial, has_aux=True)(
            state.params, batch, key)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        return TrainState(params=params, opt_state=opt_state, step=state.step + 1), stats

    print("compiling...", flush=True)
    key, kb, ks = jax.random.split(key, 3)
    batch = sample_batch(kb, sim_cfg, args.batch_size)
    state, stats = train_step(state, batch, ks)
    print("first-step stats:", {k: float(v) for k, v in jax.device_get(stats).items()})

    t0 = time.time()
    for step in range(1, args.n_steps + 1):
        key, kb, ks = jax.random.split(key, 3)
        batch = sample_batch(kb, sim_cfg, args.batch_size)
        state, stats = train_step(state, batch, ks)
        if step % args.log_every == 0:
            stats = jax.device_get(stats)
            dt = time.time() - t0
            print(f"[step {step:6d}]  t={dt:.1f}s  "
                  f"loss={float(stats['loss_total']):.3f}  "
                  f"coord={float(stats['loss_coord']):.2f}  "
                  f"score={float(stats['loss_score']):.3f}  "
                  f"knot={float(stats['loss_knot']):.3f}",
                  flush=True)
        if step % args.save_every == 0:
            (out_dir / f"ckpt_step{step:06d}.pkl").write_bytes(
                pickle.dumps(dict(params=jax.device_get(state.params),
                                    step=int(state.step),
                                    cfg=cfg.__dict__)))
            print(f"  saved checkpoint at step {step}")

    print("done.")


if __name__ == "__main__":
    main()
