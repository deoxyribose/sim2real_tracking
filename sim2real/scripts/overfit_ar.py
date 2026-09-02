"""Overfit v8 AR on a small fixed batch of sim clips (sanity check).

Sample N=8 sim clips once, freeze them, train from scratch using v8's
recipe. Track loss + rollout-eval recall on the same batch. Diagnostic
outcomes:
  loss→0 AND recall→1.0    training loop is fine, sim-recall gap = coverage/generalisation
  loss→0 BUT recall stays   loss-to-recall gap (loss doesn't drive recall)
  loss doesn't decrease     bug in loss or forward pass
"""
from __future__ import annotations

import argparse
import pickle
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from sim2real.eval_v2.coverage import _chamfer_polylines
from sim2real.model.unet_ar import (
    AttachmentHead, KnotGenerator, UNetARBackbone, UNetARConfig,
    sample_batched_noise,
)
from sim2real.scripts.ar_batched import make_sampler, sample_pool_one_clip
from sim2real.scripts.losses_ar_v2 import make_loss_fn
from sim2real.scripts.train_ar import (
    TrainState, loss_fn, _apply_photometric_augs,
)
from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip


def build_batch(key, sim_cfg: DiverseSimConfig, batch_size: int):
    """Sample a batch of clips. Returns per-clip clip_median + temporal_median +
    GT curves + valid masks + raw display images."""
    keys = jax.random.split(key, batch_size)
    outs = jax.vmap(lambda k: sample_clip(k, sim_cfg))(keys)
    clip = outs["clip_median"]                       # (B, T, H, W)
    static_med = outs["temporal_median"][..., None]  # (B, H, W, 1)
    T = clip.shape[1]
    curves = outs["curves"][:, T // 2]               # (B, N_flag, K_arc, 2)
    valid = outs["flagella"]["alive"]                # (B, N_flag)
    raw = outs["clip_raw"][:, T // 2]                # (B, H, W)
    return clip, static_med, curves, valid, raw


def rollout_recall(sampler, params, backbone, attach_head, knot_gen, cfg,
                     clips_np, smeds_np, gts_all, valids_all,
                     tta_angles=(0.0,), coverage=6.0,
                     n_draws=2, n_attach=8, n_rollouts=4):
    """Roll out on each clip, compute per-flagellum coverage recall."""
    key = jax.random.key(999)
    n_hit, n_total = 0, 0
    per_clip = []
    for b in range(clips_np.shape[0]):
        rollouts, key = sample_pool_one_clip(
            params, backbone, attach_head, knot_gen, cfg,
            clips_np[b], smeds_np[b],
            list(tta_angles), flips=(False, True),
            n_draws=n_draws, n_attach=n_attach, n_rollouts=n_rollouts,
            score_thresh=0.02, key=key, _sampler=sampler)
        hits = 0; tot = 0
        for i, g in enumerate(gts_all[b]):
            if not bool(valids_all[b, i]):
                continue
            tot += 1
            if not rollouts:
                continue
            dists = [_chamfer_polylines(rl, g) for rl in rollouts]
            if min(dists) <= coverage:
                hits += 1
        n_hit += hits; n_total += tot
        per_clip.append((hits, tot))
    return n_hit, n_total, per_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=3e-4)
    ap.add_argument("--coord-weight", type=float, default=5.0)
    ap.add_argument("--score-weight", type=float, default=5.0)
    ap.add_argument("--knot-weight", type=float, default=1.0)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--base-channels", type=int, default=48)
    ap.add_argument("--patch-size", type=int, default=12)  # v8's default
    ap.add_argument("--step-max", type=float, default=4.5)  # v8's default
    ap.add_argument("--no-augs", action="store_true")
    ap.add_argument("--no-sigma-scale", action="store_true", default=True,
                    help="v8 was trained WITHOUT σ-scale")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-seed", type=int, default=42)
    ap.add_argument("--score-mode", choices=["wide-gauss", "hard-mask", "focal"],
                    default="wide-gauss")
    ap.add_argument("--coord-mode", choices=["min-over-all", "per-cell-in-radius"],
                    default="min-over-all")
    ap.add_argument("--mask-radius-px", type=float, default=16.0)
    ap.add_argument("--knot-label-smoothing", type=float, default=0.0)
    ap.add_argument("--eval-n-draws", type=int, default=2)
    ap.add_argument("--eval-n-attach", type=int, default=8)
    ap.add_argument("--eval-n-rollouts", type=int, default=4)
    ap.add_argument("--eval-tta-angles", nargs="+", type=float, default=[0.0])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cfg = UNetARConfig(T=args.T, H=args.H, W=args.H,
                        base_channels=args.base_channels,
                        patch_size=args.patch_size,
                        step_max=args.step_max)
    sim_cfg = DiverseSimConfig(T=args.T, H=args.H, W=args.H,
                                 sigma_scale_residual=not args.no_sigma_scale)
    print(f"cfg: H={cfg.H} T={cfg.T} patch={cfg.patch_size} "
          f"step_max={cfg.step_max} n_bins=(ang{cfg.n_angle_bins}, "
          f"step{cfg.n_step_bins})   n_max_flag={sim_cfg.n_max_flagella}")

    backbone = UNetARBackbone(cfg=cfg)
    attach_head = AttachmentHead(cfg=cfg)
    knot_gen = KnotGenerator(cfg=cfg)

    # --- Sample fixed batch ---
    bkey = jax.random.key(args.batch_seed)
    clip, static_med, curves, valid, raw = build_batch(bkey, sim_cfg, args.batch_size)
    clip = jax.device_get(clip); static_med = jax.device_get(static_med)
    curves_np = np.asarray(curves); valid_np = np.asarray(valid)
    raw_np = np.asarray(raw)
    print(f"fixed batch: B={args.batch_size}  n_flag/clip: "
          f"{[int(valid_np[b].sum()) for b in range(args.batch_size)]}")

    # --- Model init ---
    key = jax.random.key(args.seed)
    key, k_b, k_a, k_k = jax.random.split(key, 4)
    dummy_video = jnp.zeros((1, cfg.T, cfg.H, cfg.W))
    dummy_noise = jnp.zeros((1, cfg.H, cfg.W, 1))
    dummy_med = jnp.zeros((1, cfg.H, cfg.W, 1))
    p_backbone = backbone.init(k_b, dummy_video, dummy_noise, dummy_med, train=True)
    _, dummy_gr = backbone.apply(p_backbone, dummy_video, dummy_noise, dummy_med,
                                     train=True)
    p_attach = attach_head.init(k_a, dummy_gr)
    p_knot = knot_gen.init(k_k, jnp.zeros((1, cfg.patch_size, cfg.patch_size,
                                              cfg.feature_channels)))
    params = dict(backbone=p_backbone, attach=p_attach, knot=p_knot)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"model params: {n_params/1e6:.2f} M")

    lr_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr,
        warmup_steps=args.warmup_steps, decay_steps=args.n_steps,
        end_value=1e-6)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_fn, weight_decay=args.weight_decay),
    )
    opt_state = optimizer.init(params)
    state = TrainState(params=params, opt_state=opt_state, step=jnp.array(0))

    lfn = make_loss_fn(score_mode=args.score_mode, coord_mode=args.coord_mode,
                          mask_radius_px=args.mask_radius_px,
                          knot_label_smoothing=args.knot_label_smoothing)
    loss_partial = partial(lfn, backbone=backbone, attach_head=attach_head,
                            knot_gen=knot_gen, cfg=cfg,
                            coord_weight=args.coord_weight,
                            score_weight=args.score_weight,
                            knot_weight=args.knot_weight)
    print(f"loss: score_mode={args.score_mode}  coord_mode={args.coord_mode}  "
          f"mask_R={args.mask_radius_px}")

    @partial(jax.jit, donate_argnums=(0,))
    def train_step(state, batch, key):
        (loss, stats), grads = jax.value_and_grad(loss_partial, has_aux=True)(
            state.params, batch, key)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        return TrainState(params=params, opt_state=opt_state, step=state.step + 1), stats

    # Bake fixed batch onto GPU once
    batch_gpu = (jnp.asarray(clip), jnp.asarray(static_med),
                  curves, valid)

    print("compiling...", flush=True)
    key, k = jax.random.split(key)
    if not args.no_augs:
        # Re-apply photometric augs stochastically per step (per v8 recipe)
        c_a, s_a = _apply_photometric_augs(k, batch_gpu[0], batch_gpu[1])
        this_batch = (c_a, s_a, batch_gpu[2], batch_gpu[3])
    else:
        this_batch = batch_gpu
    key, ks = jax.random.split(key)
    state, stats = train_step(state, this_batch, ks)
    print("first-step:", {k_: float(v) for k_, v in jax.device_get(stats).items()})

    sampler = make_sampler(cfg, backbone, attach_head, knot_gen)
    losses = []; recalls = []
    t0 = time.time()
    for step in range(1, args.n_steps + 1):
        key, k = jax.random.split(key)
        if not args.no_augs:
            c_a, s_a = _apply_photometric_augs(k, batch_gpu[0], batch_gpu[1])
            this_batch = (c_a, s_a, batch_gpu[2], batch_gpu[3])
        else:
            this_batch = batch_gpu
        key, ks = jax.random.split(key)
        state, stats = train_step(state, this_batch, ks)
        if step % args.log_every == 0:
            stats = jax.device_get(stats)
            dt = time.time() - t0
            losses.append((step, float(stats["loss_total"]),
                            float(stats["loss_coord"]),
                            float(stats["loss_score"]),
                            float(stats["loss_knot"])))
            print(f"[{step:5d}] t={dt:5.1f}s  L={float(stats['loss_total']):.3f}  "
                  f"c={float(stats['loss_coord']):.2f}  "
                  f"s={float(stats['loss_score']):.3f}  "
                  f"k={float(stats['loss_knot']):.3f}", flush=True)
        if step % args.eval_every == 0:
            # rollout recall on the fixed batch (deterministic noise each eval)
            gts_all = [[np.asarray(curves_np[b, i]) for i in range(curves_np.shape[1])]
                        for b in range(args.batch_size)]
            n_hit, n_total, per_clip = rollout_recall(
                sampler, jax.device_get(state.params),
                backbone, attach_head, knot_gen, cfg,
                clip, static_med.squeeze(-1), gts_all, valid_np,
                tta_angles=tuple(args.eval_tta_angles),
                n_draws=args.eval_n_draws,
                n_attach=args.eval_n_attach,
                n_rollouts=args.eval_n_rollouts)
            r = n_hit / max(n_total, 1)
            recalls.append((step, r))
            print(f"       eval  recall = {n_hit}/{n_total} = {r:.3f}",
                  flush=True)

    # --- Save final ckpt ---
    (out_dir / "final_params.pkl").write_bytes(
        pickle.dumps(dict(params=jax.device_get(state.params),
                            step=int(state.step), cfg=cfg.__dict__)))

    # --- Plot loss + recall ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    if losses:
        L = np.asarray(losses)
        ax[0].plot(L[:, 0], L[:, 1], label="total")
        ax[0].plot(L[:, 0], L[:, 2], label="coord")
        ax[0].plot(L[:, 0], L[:, 3], label="score")
        ax[0].plot(L[:, 0], L[:, 4], label="knot")
        ax[0].set_yscale("log"); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
        ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)
        ax[0].set_title("train loss (fixed batch)")
    if recalls:
        R = np.asarray(recalls)
        ax[1].plot(R[:, 0], R[:, 1], "-o")
        ax[1].set_xlabel("step"); ax[1].set_ylabel("recall")
        ax[1].set_ylim(0, 1.05); ax[1].grid(True, alpha=0.3)
        ax[1].set_title(f"rollout recall on same {args.batch_size} clips "
                          f"(tta_angles=(0,), 6px)")
    fig.tight_layout()
    fig.savefig(out_dir / "curve.png", dpi=140)
    print(f"wrote {out_dir}/final_params.pkl and curve.png")


if __name__ == "__main__":
    main()
