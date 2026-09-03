"""Matching stability diagnostic.

Loads a checkpoint, samples N fresh videos, forwards through the model, and per video per
frame computes the Hungarian assignment (pred slot -> GT slot, using position-only cost).
Then reports:

  - identity switch rate: fraction of (video, GT-slot) pairs where the matched pred-slot
    index changes between frame t and frame t+1. Averaged over t and alive GT slots.
  - per-frame IoU (matched_seg_iou) mean +/- std across the sampled videos.
  - per-slot assignment histogram: how often each pred slot ends up matched to each GT slot.
    A well-behaved model has near-diagonal (or at least peaky) rows; a poorly-behaved one is
    uniform (permutation-averaged).

If switches are frequent, the per-frame matching loss is being averaged across identity
flips, which explains why L_where + L_pres can look "converged" while eval IoU plateaus.

Usage:
    PYTHONPATH=. python3 -m sim2real.scripts.matching_stability \\
        --ckpt runs/overnight_v3/two_cells_baseline/ckpts/step_100000.pkl \\
        --sim two_cells --n-max 4 --n-videos 64 \\
        --glimpse-size 24 --d-model 128 --n-transformer-layers 3 --stem-strides 2 2 1
"""

from __future__ import annotations

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from sim2real.eval.fg_ari import fg_ari
from sim2real.eval.seg_iou import matched_seg_iou
from sim2real.losses.matching import build_cost_zwhere, _hungarian_cpu
from sim2real.model.model import ModelConfig, SlotVideoModel
from sim2real.sim.api import build_sim
from sim2real.train.ckpt import load as ckpt_load
from sim2real.types import SimSample


def slice_to_model(batch, Nm):
    return SimSample(
        video=batch.video,
        z_where=batch.z_where[:, :, :Nm],
        z_pres=batch.z_pres[:, :, :Nm],
        z_style=batch.z_style,
        masks=batch.masks[:, :, :Nm],
        z_what=None if batch.z_what is None else batch.z_what[:, :Nm],
        meta=batch.meta,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sim", default="two_cells")
    ap.add_argument("--n-max", type=int, required=True)
    ap.add_argument("--n-videos", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    # model config (must match ckpt)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-transformer-layers", type=int, default=3)
    ap.add_argument("--glimpse-size", type=int, default=24)
    ap.add_argument("--stem-strides", type=int, nargs="+", default=[2, 2, 1])
    ap.add_argument("--anchor-slots", action="store_true")
    ap.add_argument("--use-slot-attention", action="store_true")
    ap.add_argument("--use-neural-em", action="store_true")
    ap.add_argument("--use-isa", action="store_true")
    ap.add_argument("--isa-abs-pe-weight", type=float, default=0.0)
    ap.add_argument("--isa-anchor-init", action="store_true")
    ap.add_argument("--anchor-init-fixed", action="store_true")
    ap.add_argument("--z-what-init-std", type=float, default=0.2)
    ap.add_argument("--nem-attn-temp", type=float, default=1.0)
    ap.add_argument("--nem-use-bg-slot", action="store_true")
    ap.add_argument("--nem-dual-source-what", action="store_true")
    ap.add_argument("--nem-use-pca-theta", action="store_true")
    ap.add_argument("--decoder-use-film", action="store_true")
    ap.add_argument("--nem-theta-delta-scale", type=float, default=0.05)
    ap.add_argument("--hard-pres-gate", action="store_true",
                    help="Match with hard pres gate: dead pred slots can't win alive-GT columns.")
    ap.add_argument("--savi-bootstrap", action="store_true",
                    help="Pass GT z_where[:, 0] as slot init per video (required if the model "
                         "was trained with --savi-bootstrap, else slot init is random and the "
                         "model behaves as untrained).")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(args.ckpt)

    # Load ckpt + build model.
    ck = ckpt_load(args.ckpt)
    params = ck["params"]
    model_cfg = ModelConfig(
        n_max=args.n_max, d_model=args.d_model, n_heads=4,
        n_transformer_layers=args.n_transformer_layers,
        z_what_dim=64, z_style_dim=4, glimpse_size=args.glimpse_size,
        stem_channels=(16, 32, 64), stem_strides=tuple(args.stem_strides),
        n_groups=1, use_background=True, bg_base_res=4, bg_channels=(8,),
        anchor_slots=args.anchor_slots, use_slot_attention=args.use_slot_attention,
        use_neural_em=args.use_neural_em, use_isa=args.use_isa,
        isa_abs_pe_weight=args.isa_abs_pe_weight, isa_anchor_init=args.isa_anchor_init,
        anchor_init_fixed=args.anchor_init_fixed, z_what_init_std=args.z_what_init_std,
        nem_attn_temp=args.nem_attn_temp,
        nem_use_bg_slot=args.nem_use_bg_slot,
        nem_dual_source_what=args.nem_dual_source_what,
        nem_use_pca_theta=args.nem_use_pca_theta,
        decoder_use_film=args.decoder_use_film,
        nem_theta_delta_scale=args.nem_theta_delta_scale,
    )
    model = SlotVideoModel(cfg=model_cfg)

    # Sample fresh videos.
    batch_fn, _ = build_sim(args.sim)
    key = jax.random.key(args.seed)
    key, kb = jax.random.split(key)
    batch = jax.jit(lambda k: batch_fn(k, args.n_videos))(kb)
    bm = slice_to_model(batch, args.n_max)
    T = int(bm.video.shape[1])
    N = args.n_max
    print(f"loaded {args.ckpt} (step={ck.get('step','?')}); "
          f"batch: {bm.video.shape}, N={N}, T={T}", flush=True)

    # Forward through the model in chunks (avoid OOM on B=64 with vmap over full batch).
    CHUNK = 4

    if args.savi_bootstrap:
        eval_boot0 = bm.z_where[:, 0]  # (B, N, 5)
    else:
        eval_boot0 = None

    @jax.jit
    def fwd(video, k, boot0):
        keys = jax.random.split(k, video.shape[0])
        def one(v, kk, b):
            return model.apply(params, v, kk, bootstrap_zwhere0=b)
        in_axes = (0, 0, 0 if boot0 is not None else None)
        return jax.vmap(one, in_axes=in_axes)(video, keys, boot0)

    key, kf = jax.random.split(key)
    zw_chunks, zp_chunks, m_chunks = [], [], []
    for i in range(0, args.n_videos, CHUNK):
        chunk = bm.video[i:i + CHUNK]
        boot_chunk = eval_boot0[i:i + CHUNK] if eval_boot0 is not None else None
        kf, kc = jax.random.split(kf)
        out_c = fwd(chunk, kc, boot_chunk)
        zw_chunks.append(np.asarray(out_c.z_where))
        zp_chunks.append(np.asarray(out_c.z_pres))
        m_chunks.append(np.asarray(out_c.masks_pred))
    pred_zw = np.concatenate(zw_chunks, axis=0)
    pred_zp = np.concatenate(zp_chunks, axis=0)
    pred_m  = np.concatenate(m_chunks, axis=0)
    gt_zw   = np.asarray(bm.z_where)       # (B, T, N, 5)
    gt_zp   = np.asarray(bm.z_pres)        # (B, T, N)
    gt_m    = np.asarray(bm.masks)         # (B, T, N, H, W)

    # matching_stability.py uses ONLY position — same as build_cost_zwhere.
    # Compute per-frame perm for each video via scipy (CPU).
    def perm_one(pred_zw_v, gt_zw_v, gt_zp_v):
        # perm shape (T, N); perm[t][j] = pred slot matched to GT slot j at frame t
        cost = build_cost_zwhere(jnp.asarray(pred_zw_v), jnp.asarray(gt_zw_v),
                                 jnp.asarray(gt_zp_v))
        return np.asarray(cost)

    switch_rates = []       # per video
    per_video_iou = []
    per_video_ari = []
    assign_hist = np.zeros((N, N), dtype=np.int64)  # rows=pred slot, cols=GT slot

    for v in range(args.n_videos):
        # Per-frame Hungarian
        perms = np.zeros((T, N), dtype=np.int32)
        for t in range(T):
            cost = np.asarray(build_cost_zwhere(
                jnp.asarray(pred_zw[v, t]),
                jnp.asarray(gt_zw[v, t]),
                jnp.asarray(gt_zp[v, t]),
                pred_pres=jnp.asarray(pred_zp[v, t]) if args.hard_pres_gate else None,
                hard_pres_gate=args.hard_pres_gate,
            ))
            perms[t] = _hungarian_cpu(cost)  # length N, perms[t][j] = pred idx for GT j

        # Assignment histogram (alive GT only)
        for t in range(T):
            for j in range(N):
                if gt_zp[v, t, j] > 0.5:
                    assign_hist[perms[t, j], j] += 1

        # Identity switch rate: for each alive GT j and pair (t, t+1), does perms[t][j] change?
        n_pairs, n_switches = 0, 0
        for t in range(T - 1):
            for j in range(N):
                if gt_zp[v, t, j] > 0.5 and gt_zp[v, t + 1, j] > 0.5:
                    n_pairs += 1
                    if perms[t, j] != perms[t + 1, j]:
                        n_switches += 1
        sw = n_switches / max(n_pairs, 1)
        switch_rates.append(sw)

        # Per-video FG-ARI (primary segmentation metric).
        ari = fg_ari(pred_m[v], gt_m[v], gt_zp[v])
        if ari == ari:
            per_video_ari.append(ari)
        # Legacy matched-seg-IoU for continuity.
        iou = float(matched_seg_iou(
            jnp.asarray(pred_zw[v]), jnp.asarray(pred_m[v]),
            jnp.asarray(gt_zw[v]), jnp.asarray(gt_m[v]), jnp.asarray(gt_zp[v]),
            pred_pres=jnp.asarray(pred_zp[v]) if args.hard_pres_gate else None,
            hard_pres_gate=args.hard_pres_gate,
        ))
        per_video_iou.append(iou)

    switch_rates = np.asarray(switch_rates)
    per_video_iou = np.asarray(per_video_iou)
    per_video_ari = np.asarray(per_video_ari) if per_video_ari else np.array([np.nan])

    # Row-normalize the histogram for readability
    row_sums = assign_hist.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    assign_freq = assign_hist / row_sums

    summary = {
        "ckpt": args.ckpt,
        "n_videos": int(args.n_videos),
        "T": int(T),
        "N": int(N),
        "switch_rate_mean": float(switch_rates.mean()),
        "switch_rate_std":  float(switch_rates.std()),
        "switch_rate_min":  float(switch_rates.min()),
        "switch_rate_max":  float(switch_rates.max()),
        "fg_ari_mean": float(per_video_ari.mean()),
        "fg_ari_std":  float(per_video_ari.std()),
        "iou_legacy_mean": float(per_video_iou.mean()),
        "iou_legacy_std":  float(per_video_iou.std()),
        "assign_hist_counts": assign_hist.tolist(),
        "assign_hist_row_normalized": assign_freq.tolist(),
    }

    out_path = os.path.join(out_dir, "matching_stability.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Matching stability on {args.n_videos} fresh {args.sim} videos ===")
    print(f"identity switch rate  mean={switch_rates.mean():.3f} "
          f"std={switch_rates.std():.3f} min={switch_rates.min():.3f} "
          f"max={switch_rates.max():.3f}")
    print(f"FG-ARI                mean={per_video_ari.mean():.3f} "
          f"std={per_video_ari.std():.3f}")
    print(f"IoU (legacy)          mean={per_video_iou.mean():.3f} "
          f"std={per_video_iou.std():.3f}")
    print(f"\nassignment histogram (rows=pred slot, cols=GT slot; counts across all "
          f"(video, frame, alive-GT-slot)):")
    print(np.array2string(assign_hist, max_line_width=120))
    print(f"\nrow-normalized (how each pred slot distributes across GT slots):")
    print(np.array2string(assign_freq, precision=2, suppress_small=True, max_line_width=120))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
