"""Eval the AR knot generator.

For each real annotation:
  1. Canonicalize + resize to model input
  2. Predict attachment grid → sample top-K attachments (by score)
  3. For each attachment: N noise draws × K autoregressive knot rollouts →
     each rollout samples per-knot polar step from the categorical
     distributions (with temperature).
  4. Recall = fraction of GT flagella whose min-Chamfer to any rollout is
     within `coverage_thresh` canonical px.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from sim2real.data.canonicalize import CANONICAL_H, CANONICAL_W
from sim2real.eval_v2.coverage import (
    _chamfer_polylines, canonicalize_real_frame, gt_polyline_to_canonical,
    load_real_annotations,
)
from sim2real.model.unet_ar import (
    AttachmentHead, KnotGenerator, UNetARBackbone, UNetARConfig,
    rotated_patch_batched, sample_batched_noise, unpack_attachment,
)


def load_ckpt(path: str) -> tuple[dict, UNetARConfig]:
    d = pickle.loads(Path(path).read_bytes())
    cfg = UNetARConfig(**{k: v for k, v in d["cfg"].items()
                          if k in UNetARConfig.__dataclass_fields__})
    return d["params"], cfg


def sample_attachments(f: dict, n_top: int, cfg: UNetARConfig) -> tuple[np.ndarray, np.ndarray]:
    """Given the unpacked attachment head output for one image, return the
    top `n_top` attachment (y, x) positions + their scores."""
    ys = np.arange(cfg.grid_h) * cfg.grid_stride + cfg.grid_stride / 2
    xs = np.arange(cfg.grid_w) * cfg.grid_stride + cfg.grid_stride / 2
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    cell_centers = np.stack([yy, xx], -1)                     # (gh, gw, 2)
    dy = np.asarray(f["dy"]); dx = np.asarray(f["dx"])
    pos = cell_centers[:, :, None, :] + np.stack([dy, dx], -1)  # (gh, gw, n_sug, 2)
    scores = np.asarray(jax.nn.sigmoid(f["score"]))            # (gh, gw, n_sug)
    flat_pos = pos.reshape(-1, 2)
    flat_score = scores.reshape(-1)
    idx = np.argsort(-flat_score)[:n_top]
    return flat_pos[idx].astype(np.float32), flat_score[idx].astype(np.float32)


def rollout_one_flagellum(feature_map: jnp.ndarray, params_knot: dict,
                            knot_gen: KnotGenerator, cfg: UNetARConfig,
                            attach: jnp.ndarray, key: jax.Array,
                            temperature: float = 1.0) -> jnp.ndarray:
    """Sample a single (K+1, 2) skeleton starting from `attach`."""
    K = cfg.n_knots
    def step_fn(carry, _):
        pos, tangent, k = carry
        patch = rotated_patch_batched(feature_map, pos[None], tangent[None],
                                        cfg.patch_size)         # (1, P, P, C)
        angle_logits, step_logits = knot_gen.apply(params_knot, patch)
        # Sample from categorical (with temperature)
        k_a, k_s = jax.random.fold_in(key, k), jax.random.fold_in(key, k + 100000)
        a_bin = jax.random.categorical(k_a, angle_logits[0] / temperature)
        s_bin = jax.random.categorical(k_s, step_logits[0] / temperature)
        d_ang = cfg.angle_bin_centers[a_bin]
        d_step = cfg.step_bin_centers[s_bin]
        new_tan = tangent + d_ang
        new_pos = jnp.stack([pos[0] + d_step * jnp.sin(new_tan),
                              pos[1] + d_step * jnp.cos(new_tan)])
        return (new_pos, new_tan, k + 1), new_pos
    (_, _, _), positions = jax.lax.scan(
        step_fn, (attach, jnp.array(0.0), jnp.int32(0)), jnp.arange(K))
    return jnp.concatenate([attach[None], positions], axis=0)


def rollout_batch(feature_map: jnp.ndarray, params_knot: dict,
                    knot_gen: KnotGenerator, cfg: UNetARConfig,
                    attaches: jnp.ndarray, keys: jax.Array,
                    temperature: float = 1.0) -> jnp.ndarray:
    """Rollout a skeleton for each (attachment, key) pair. Returns (M, K+1, 2)."""
    return jax.vmap(
        lambda a, k: rollout_one_flagellum(feature_map, params_knot, knot_gen,
                                            cfg, a, k, temperature))(attaches, keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-draws", type=int, default=8, help="noise draws for backbone")
    ap.add_argument("--n-attach", type=int, default=16, help="top-N attachments to keep per draw")
    ap.add_argument("--n-rollouts", type=int, default=4,
                    help="AR rollouts per attachment (with different sampling seeds)")
    ap.add_argument("--score-thresh", type=float, default=0.03)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--coverage-thresh", type=float, default=6.0,
                    help="in MODEL-canvas pixels; 6 @ 128 = 12 @ 256")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params, cfg = load_ckpt(args.ckpt)
    backbone = UNetARBackbone(cfg=cfg)
    attach_head = AttachmentHead(cfg=cfg)
    knot_gen = KnotGenerator(cfg=cfg)

    annots = load_real_annotations()
    if args.limit > 0: annots = annots[: args.limit]
    print(f"n annotations: {len(annots)}", flush=True)

    key = jax.random.key(42)
    all_pre_recalls, all_n_gt = [], []
    per_ann = []

    for ai, ann in enumerate(annots):
        try:
            canon, cfg_can = canonicalize_real_frame(ann["meta"],
                                                       ann["src_width_px"], T=cfg.T)
        except Exception as e:
            print(f"  [{ai:3d}] skip {ann['name']}: {e}", flush=True); continue
        clip = canon["clip"]; smed_native = canon["static_median"].astype(np.float32) / 255
        src_h, src_w = clip.shape[1], clip.shape[2]
        scale_y = cfg.H / src_h; scale_x = cfg.W / src_w
        clip_r = np.stack([cv2.resize(clip[t], (cfg.W, cfg.H),
                                        interpolation=cv2.INTER_AREA)
                            for t in range(clip.shape[0])], axis=0).astype(np.float32)
        smed_r = cv2.resize(smed_native, (cfg.W, cfg.H),
                             interpolation=cv2.INTER_AREA).astype(np.float32)

        gt_curves = []
        for pl in ann["gt_polylines_native"]:
            gc = gt_polyline_to_canonical(pl, ann["meta"], cfg_can,
                                            canonical_h=CANONICAL_H,
                                            canonical_w=CANONICAL_W)
            if len(gc) >= 4:
                gt_curves.append(gc * np.asarray([scale_y, scale_x]))
        if not gt_curves:
            continue

        all_rollouts = []
        video = jnp.asarray(clip_r)[None]
        smed = jnp.asarray(smed_r)[None, ..., None]
        for _ in range(args.n_draws):
            key, k = jax.random.split(key)
            noise = sample_batched_noise(k, 1, cfg)
            full_res, grid = backbone.apply(params["backbone"], video, noise,
                                              smed, train=False)
            attach = attach_head.apply(params["attach"], grid)
            f = unpack_attachment(attach[0])
            attaches, scores = sample_attachments(f, args.n_attach, cfg)
            # Only keep attachments above score threshold
            keep = scores >= args.score_thresh
            attaches = attaches[keep]
            if len(attaches) == 0: continue
            for r in range(args.n_rollouts):
                key, kr = jax.random.split(key)
                rollout_keys = jax.random.split(kr, len(attaches))
                rollouts = rollout_batch(full_res[0], params["knot"], knot_gen,
                                          cfg, jnp.asarray(attaches),
                                          rollout_keys, args.temperature)
                all_rollouts.extend([np.asarray(r) for r in rollouts])

        if not all_rollouts:
            covered = [False] * len(gt_curves)
            all_dists = [float("inf")] * len(gt_curves)
        else:
            covered, all_dists = [], []
            for gt in gt_curves:
                dists = [_chamfer_polylines(r, gt) for r in all_rollouts]
                m = min(dists)
                covered.append(m <= args.coverage_thresh)
                all_dists.append(m)
        recall = float(sum(covered) / len(covered))
        all_pre_recalls.append(recall); all_n_gt.append(len(gt_curves))
        per_ann.append(dict(name=ann["name"], n_gt=len(gt_curves),
                              n_rollouts=len(all_rollouts),
                              pre_recall=recall,
                              min_chamfers=[float(x) for x in all_dists]))
        print(f"  [{ai:3d}] {ann['name']:20s}  n_gt={len(gt_curves)}  "
              f"n_roll={len(all_rollouts)}  recall={recall:.2f}  "
              f"min_d={[round(x,1) for x in all_dists]}",
              flush=True)

    total = sum(r * n for r, n in zip(all_pre_recalls, all_n_gt)) / \
             max(sum(all_n_gt), 1)
    print(f"\nOVERALL pre_recall = {total:.3f} @ {args.coverage_thresh}px")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(
        ckpt=args.ckpt,
        pre_recall=float(total),
        coverage_thresh=args.coverage_thresh,
        n_annotations=len(annots),
        per_annotation=per_ann,
    ), indent=2))


if __name__ == "__main__":
    main()
