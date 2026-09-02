"""Per-knot angle/step entropy — measure model determinism across rollouts.

For each clip (a) run the model, (b) pick top-K attachments, (c) at each
attachment roll a knot forward: at each of the 24 steps record the
per-knot angle_logits and step_logits (this is the categorical the model
uses to sample). Average their entropies across all attachments.

Report:
  - Per-knot mean angle entropy (24 numbers)
  - Per-knot mean step entropy  (24 numbers)
  - Compare real vs sim.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from sim2real.eval_v2.coverage import (
    canonicalize_real_frame, load_real_annotations,
)
from sim2real.model.unet_ar import (
    AttachmentHead, KnotGenerator, UNetARBackbone,
    rotated_patch_batched, sample_batched_noise, unpack_attachment,
)
from sim2real.scripts.eval_ar import load_ckpt, sample_attachments
from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip


def entropy(logits):
    """Categorical entropy in nats. logits: (..., n_bins)."""
    p = jax.nn.softmax(logits, axis=-1)
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(p * logp, axis=-1)


def rollout_with_logits(feature_map, params_knot, knot_gen, cfg, attach, key,
                          temperature=1.0):
    """Like rollout_one_flagellum but also records per-knot logits."""
    K = cfg.n_knots
    def step_fn(carry, _):
        pos, tangent, k = carry
        patch = rotated_patch_batched(feature_map, pos[None], tangent[None],
                                        cfg.patch_size)
        angle_logits, step_logits = knot_gen.apply(params_knot, patch)
        a_bin = jax.random.categorical(jax.random.fold_in(key, k),
                                          angle_logits[0] / temperature)
        s_bin = jax.random.categorical(jax.random.fold_in(key, k + 100000),
                                          step_logits[0] / temperature)
        d_ang = cfg.angle_bin_centers[a_bin]
        d_step = cfg.step_bin_centers[s_bin]
        new_tan = tangent + d_ang
        new_pos = jnp.stack([pos[0] + d_step * jnp.sin(new_tan),
                              pos[1] + d_step * jnp.cos(new_tan)])
        return (new_pos, new_tan, k + 1), (angle_logits[0], step_logits[0])
    (_, _, _), (a_logits, s_logits) = jax.lax.scan(
        step_fn, (attach, jnp.array(0.0), jnp.int32(0)), jnp.arange(K))
    return a_logits, s_logits    # (K, n_angle_bins), (K, n_step_bins)


def run_one_source(source_name, clips_iter, params, backbone, attach_head,
                     knot_gen, cfg, n_top_attach=8):
    """clips_iter: iterable of (clip_r, smed_r) numpy arrays."""
    key = jax.random.key(42)
    all_a_ent, all_s_ent = [], []
    for i, (clip, smed) in enumerate(clips_iter):
        video = jnp.asarray(clip)[None]
        smed_in = jnp.asarray(smed)[None, ..., None]
        key, k = jax.random.split(key)
        noise = sample_batched_noise(k, 1, cfg)
        full_res, grid = backbone.apply(params["backbone"], video, noise,
                                          smed_in, train=False)
        attach = attach_head.apply(params["attach"], grid)
        f = unpack_attachment(attach[0])
        atts, scores = sample_attachments(f, n_top_attach, cfg)
        if len(atts) == 0: continue
        for att in atts:
            a_l, s_l = rollout_with_logits(full_res[0], params["knot"], knot_gen,
                                              cfg, jnp.asarray(att), key)
            all_a_ent.append(np.asarray(entropy(a_l)))
            all_s_ent.append(np.asarray(entropy(s_l)))
        if (i + 1) % 5 == 0:
            print(f"  {source_name}  [{i+1}]", flush=True)
    return np.stack(all_a_ent), np.stack(all_s_ent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-clips", type=int, default=24)
    ap.add_argument("--n-attach", type=int, default=8)
    ap.add_argument("--sim-seed", type=int, default=1234)
    ap.add_argument("--sigma-scale", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params, cfg = load_ckpt(args.ckpt)
    backbone = UNetARBackbone(cfg=cfg)
    attach_head = AttachmentHead(cfg=cfg)
    knot_gen = KnotGenerator(cfg=cfg)

    # --- Real clips ---
    def real_iter():
        annots = load_real_annotations()[: args.n_clips]
        for ann in annots:
            try:
                canon, _ = canonicalize_real_frame(ann["meta"],
                                                     ann["src_width_px"], T=cfg.T)
            except Exception:
                continue
            clip = canon["clip"]
            smed = canon["static_median"].astype(np.float32) / 255
            src_h, src_w = clip.shape[1], clip.shape[2]
            clip_r = np.stack([cv2.resize(clip[t], (cfg.W, cfg.H),
                                            interpolation=cv2.INTER_AREA)
                                for t in range(clip.shape[0])], axis=0).astype(np.float32)
            smed_r = cv2.resize(smed, (cfg.W, cfg.H),
                                 interpolation=cv2.INTER_AREA).astype(np.float32)
            yield clip_r, smed_r

    # --- Sim clips ---
    def sim_iter():
        sim_cfg = DiverseSimConfig(T=cfg.T, H=cfg.H, W=cfg.W,
                                     sigma_scale_residual=args.sigma_scale)
        skey = jax.random.key(args.sim_seed)
        for _ in range(args.n_clips):
            skey, k = jax.random.split(skey)
            out = sample_clip(k, sim_cfg)
            yield np.asarray(out["clip_median"]), np.asarray(out["temporal_median"])

    print("Real..."); a_r, s_r = run_one_source("real", real_iter(),
                                                    params, backbone, attach_head,
                                                    knot_gen, cfg,
                                                    n_top_attach=args.n_attach)
    print("Sim..."); a_s, s_s = run_one_source("sim", sim_iter(),
                                                   params, backbone, attach_head,
                                                   knot_gen, cfg,
                                                   n_top_attach=args.n_attach)
    K = cfg.n_knots
    ang_max = np.log(cfg.n_angle_bins)
    step_max = np.log(cfg.n_step_bins)
    a_r_mean = a_r.mean(0); a_r_std = a_r.std(0)
    a_s_mean = a_s.mean(0); a_s_std = a_s.std(0)
    s_r_mean = s_r.mean(0); s_r_std = s_r.std(0)
    s_s_mean = s_s.mean(0); s_s_std = s_s.std(0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ks = np.arange(K)
    ax = axes[0]
    ax.fill_between(ks, a_r_mean - a_r_std, a_r_mean + a_r_std,
                     color="#4a86e8", alpha=0.15)
    ax.plot(ks, a_r_mean, "-o", color="#4a86e8", label="real", markersize=3)
    ax.fill_between(ks, a_s_mean - a_s_std, a_s_mean + a_s_std,
                     color="#e69138", alpha=0.15)
    ax.plot(ks, a_s_mean, "-o", color="#e69138", label="sim", markersize=3)
    ax.axhline(ang_max, color="gray", lw=0.6, linestyle="--",
                label=f"uniform ({ang_max:.2f})")
    ax.set_xlabel("knot index"); ax.set_ylabel("entropy (nats)")
    ax.set_title(f"Δangle entropy per knot  (max = log({cfg.n_angle_bins}))")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.fill_between(ks, s_r_mean - s_r_std, s_r_mean + s_r_std,
                     color="#4a86e8", alpha=0.15)
    ax.plot(ks, s_r_mean, "-o", color="#4a86e8", label="real", markersize=3)
    ax.fill_between(ks, s_s_mean - s_s_std, s_s_mean + s_s_std,
                     color="#e69138", alpha=0.15)
    ax.plot(ks, s_s_mean, "-o", color="#e69138", label="sim", markersize=3)
    ax.axhline(step_max, color="gray", lw=0.6, linestyle="--",
                label=f"uniform ({step_max:.2f})")
    ax.set_xlabel("knot index"); ax.set_ylabel("entropy (nats)")
    ax.set_title(f"Δstep entropy per knot  (max = log({cfg.n_step_bins}))")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Per-knot AR entropy (real vs sim, v8 AR ckpt), "
                 f"{args.n_clips} clips × {args.n_attach} attach", fontsize=11)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")
    print()
    print("summary (mean over knots):")
    print(f"  angle  real={a_r_mean.mean():.2f}   sim={a_s_mean.mean():.2f}"
          f"   uniform={ang_max:.2f}")
    print(f"  step   real={s_r_mean.mean():.2f}   sim={s_s_mean.mean():.2f}"
          f"   uniform={step_max:.2f}")


if __name__ == "__main__":
    main()
