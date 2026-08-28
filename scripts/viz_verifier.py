"""Visualize verifier selections on raw source frames.

For each frame:
  1. Model forward → sample candidates.
  2. Verifier solves → returns selected flag/cell indices.
  3. Overlay: green = GT flagellum, cyan = selected cell, red = selected flagellum,
     dim gray = rejected candidates.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import tifffile
from PIL import Image

from sim2real.data import CANONICAL_H, CANONICAL_W, CLASS_FLAGELLUM
from sim2real.eval_v2.coverage import canonicalize_real_frame, load_real_annotations
from sim2real.model_v2 import DETRSlotModel
from sim2real.verifier import build_and_solve, VerifyConfig
from sim2real.verifier.run_verify import build_candidate_pools

# Inline the helpers from viz_detr_slot_raw.py so we don't need `scripts` as a package.
def canonical_pts_to_native_cropped(pts_canonical, meta, cfg_c):
    scale = cfg_c.resample_scale
    cropped_h, cropped_w = meta["orig_h"], meta["orig_w"]
    new_h = int(round(cropped_h * scale))
    new_w = int(round(cropped_w * scale))
    pad_top = (CANONICAL_H - new_h) // 2
    pad_left = (CANONICAL_W - new_w) // 2
    pts = pts_canonical.astype(np.float32).copy()
    pts[..., 0] -= pad_top
    pts[..., 1] -= pad_left
    pts /= scale
    return pts


def native_cropped_to_full(pts, meta):
    pts = pts.copy()
    pts[..., 0] += meta["crop_top"]
    return pts


def load_raw_source(src_path):
    if src_path.endswith(".tif"):
        img = tifffile.imread(src_path)
    else:
        img = np.array(Image.open(src_path))
    if img.ndim == 3:
        img = img.mean(axis=-1)
    return img


def _draw_polyline(img_rgb, pts, color, thickness=1, alpha=1.0):
    pts_xy = np.stack([pts[:, 1], pts[:, 0]], axis=1).astype(np.int32)
    tmp = img_rgb.copy()
    for i in range(len(pts_xy) - 1):
        cv2.line(tmp, tuple(pts_xy[i]), tuple(pts_xy[i + 1]), color, thickness, lineType=cv2.LINE_AA)
    img_rgb[:] = (alpha * tmp + (1 - alpha) * img_rgb).astype(np.uint8)


def _norm(x, lo=None, hi=None):
    if lo is None:
        lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def canonical_center_to_raw(center_can, radius_can, meta, cfg_c):
    """Map a canonical (cy, cx, r) to raw pixel space."""
    scale = cfg_c.resample_scale
    cropped_h, cropped_w = meta["orig_h"], meta["orig_w"]
    new_h = int(round(cropped_h * scale))
    new_w = int(round(cropped_w * scale))
    pad_top = (CANONICAL_H - new_h) // 2
    pad_left = (CANONICAL_W - new_w) // 2
    cy = (center_can[0] - pad_top) / scale + meta["crop_top"]
    cx = (center_can[1] - pad_left) / scale
    r_raw = radius_can / scale
    return cy, cx, r_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/frans/sim2real_tracking/runs/detr_slot/v2/ckpt_step_5000.pkl")
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--scoring", default="energy")
    ap.add_argument("--max-flag", type=int, default=2)
    ap.add_argument("--max-cell", type=int, default=1)
    ap.add_argument("--birth-flag", type=float, default=1.5)
    ap.add_argument("--birth-cell", type=float, default=2.0)
    ap.add_argument("--attach-slack", type=float, default=15.0)
    ap.add_argument("--n-frames", type=int, default=12)
    ap.add_argument("--out-dir", default="/home/frans/sim2real_tracking/runs/detr_slot/v2/viz_verify")
    args = ap.parse_args()

    ckpt = pickle.load(open(args.ckpt, "rb"))
    model = DETRSlotModel(cfg=ckpt["model_cfg"])
    ann = load_real_annotations()

    seen = {}
    picks = []
    for e in ann:
        s = e["meta"]["sequence"]
        if seen.get(s, 0) < 1:
            picks.append(e); seen[s] = 1
        if len(picks) >= args.n_frames:
            break

    @jax.jit
    def fwd(p, c, e, k):
        return model.apply(p, c, e, rngs={"slots": k})

    vcfg = VerifyConfig(
        max_flagella=args.max_flag, max_cells=args.max_cell,
        attach_slack_px=args.attach_slack,
        birth_prior_per_flag=args.birth_flag, birth_prior_per_cell=args.birth_cell,
        scoring=args.scoring, solver_time_limit_s=10,
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(0)
    for i, entry in enumerate(picks):
        canon, cfg_c = canonicalize_real_frame(entry["meta"], entry["src_width_px"], T=16)
        clip = jnp.asarray(canon["clip"])[None]
        energy = jnp.asarray(canon["energy"])[None]
        key, sub = jax.random.split(key)
        out = fwd(ckpt["params"], clip, energy, sub)
        key, sub2 = jax.random.split(key)
        flag_d, cell_d, _ = build_candidate_pools(out, args.n_samples, 1.0, sub2, 0.0, 0.0)
        target = canon["clip"].min(axis=0)
        res = build_and_solve(flag_d, cell_d, target, cfg=vcfg, energy_map=canon["energy"])

        raw = load_raw_source(entry["source"])
        rgb = np.stack([_norm(raw)] * 3, axis=-1)

        # Draw REJECTED flagellum candidates in faint gray (only sample a few)
        rej_ids = [i for i in range(flag_d["renders"].shape[0]) if i not in res["selected_flag_idx"]]
        rej_sample = np.random.default_rng(0).choice(rej_ids, min(20, len(rej_ids)), replace=False) if rej_ids else []
        for k in rej_sample:
            pts_raw = native_cropped_to_full(
                canonical_pts_to_native_cropped(flag_d["pts"][k], entry["meta"], cfg_c), entry["meta"])
            _draw_polyline(rgb, pts_raw, (100, 100, 100), thickness=1, alpha=0.15)

        # Draw SELECTED cells in cyan
        for j in res["selected_cell_idx"]:
            cy_raw, cx_raw, r_raw = canonical_center_to_raw(
                cell_d["centers"][j], cell_d["radii"][j], entry["meta"], cfg_c)
            cv2.circle(rgb, (int(round(cx_raw)), int(round(cy_raw))), int(round(r_raw)),
                       (0, 220, 220), 2, cv2.LINE_AA)

        # Draw SELECTED flagellum candidates in bright red
        for k in res["selected_flag_idx"]:
            pts_raw = native_cropped_to_full(
                canonical_pts_to_native_cropped(flag_d["pts"][k], entry["meta"], cfg_c), entry["meta"])
            _draw_polyline(rgb, pts_raw, (255, 60, 60), thickness=3, alpha=1.0)
            # attachment as a small yellow dot
            ay, ax = int(round(pts_raw[0, 0])), int(round(pts_raw[0, 1]))
            cv2.circle(rgb, (ax, ay), 4, (255, 255, 60), -1)

        # GT last (topmost) in bright green
        for g in entry["gt_polylines_native"]:
            g_full = native_cropped_to_full(g.astype(np.float32), entry["meta"])
            _draw_polyline(rgb, g_full, (0, 255, 0), thickness=2, alpha=1.0)

        seq = entry["meta"]["sequence"].split("/")[-1][:35]
        cv2.putText(rgb, seq, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(rgb, f"sel: {len(res['selected_flag_idx'])}f + {len(res['selected_cell_idx'])}c",
                    (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        Image.fromarray(rgb).save(Path(args.out_dir) / f"verify_{i:02d}_{entry['name']}")
        print(f"  {i}: {entry['meta']['sequence'][-40:]} sel={len(res['selected_flag_idx'])}f {len(res['selected_cell_idx'])}c")

    print(f"[save] {args.out_dir}/")


if __name__ == "__main__":
    main()
