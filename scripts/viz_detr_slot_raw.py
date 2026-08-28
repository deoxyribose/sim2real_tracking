"""Same as viz_detr_slot.py but overlays predictions on the RAW source frame
(not the canonicalized clip midpoint). Predictions are mapped back to raw pixel
coords via inverse of the canonicalize pipeline.
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

from sim2real.data import CANONICAL_H, CANONICAL_W, CANONICAL_TARGET_WIDTH_PX, CLASS_FLAGELLUM
from sim2real.eval_v2.coverage import (
    canonicalize_real_frame, load_real_annotations,
)
from sim2real.model_v2 import DETRSlotModel, sample_flagellum_from_head


SLOT_COLORS = [
    (255, 60, 60), (60, 200, 60), (60, 120, 255), (255, 200, 30),
    (200, 60, 255), (30, 220, 220), (255, 120, 60), (150, 220, 60),
]


def canonical_pts_to_native_cropped(pts_canonical: np.ndarray, meta: dict, cfg_c) -> np.ndarray:
    """Invert canonicalize spatial transform to go from canonical (yc, xc) back to native
    cropped-frame (y, x). NOTE: still in the sampler's banner-cropped space (not raw uncropped)."""
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


def native_cropped_to_full(pts: np.ndarray, meta: dict) -> np.ndarray:
    """Add back the crop_top offset so pts are in RAW image coords."""
    pts = pts.copy()
    pts[..., 0] += meta["crop_top"]
    return pts


def load_raw_source(src_path: str) -> np.ndarray:
    if src_path.endswith(".tif"):
        img = tifffile.imread(src_path)
    else:
        img = np.array(Image.open(src_path))
    if img.ndim == 3:
        img = img.mean(axis=-1)
    return img


def _draw_polyline(img_rgb, pts, color, thickness=1, alpha=1.0):
    """In-place; pts in (y, x) format."""
    H, W = img_rgb.shape[:2]
    pts_xy = np.stack([pts[:, 1], pts[:, 0]], axis=1).astype(np.int32)
    tmp = img_rgb.copy()
    for i in range(len(pts_xy) - 1):
        cv2.line(tmp, tuple(pts_xy[i]), tuple(pts_xy[i + 1]), color, thickness, lineType=cv2.LINE_AA)
    img_rgb[:] = (alpha * tmp + (1 - alpha) * img_rgb).astype(np.uint8)


def _norm(x, lo=None, hi=None):
    if lo is None:
        lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/frans/sim2real_tracking/runs/detr_slot/v1/ckpt_step_20000.pkl")
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--sample-alpha", type=float, default=0.5)
    ap.add_argument("--mean-alpha", type=float, default=1.0)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--out-dir", default="/home/frans/sim2real_tracking/runs/detr_slot/v1/viz_raw")
    args = ap.parse_args()

    ckpt = pickle.load(open(args.ckpt, "rb"))
    model = DETRSlotModel(cfg=ckpt["model_cfg"])
    ann = load_real_annotations()
    seen = {}
    picks = []
    for e in ann:
        s = e["meta"]["sequence"]
        if seen.get(s, 0) < 1:
            picks.append(e)
            seen[s] = 1
        if len(picks) >= args.n_frames:
            break

    @jax.jit
    def fwd(p, c, e, k):
        return model.apply(p, c, e, rngs={"slots": k})

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(0)
    tiles = []
    for i, entry in enumerate(picks):
        canon, cfg_c = canonicalize_real_frame(entry["meta"], entry["src_width_px"], T=16)
        clip = jnp.asarray(canon["clip"])[None]
        energy = jnp.asarray(canon["energy"])[None]
        key, sub = jax.random.split(key)
        out = fwd(ckpt["params"], clip, energy, sub)
        key, sub2 = jax.random.split(key)
        samples = sample_flagellum_from_head(sub2, out, n_samples=args.n_samples, temperature=1.0)
        pts_samples = np.asarray(samples["pts_samples"])[0]  # (S, N, K+1, 2) canonical
        pts_mean = np.asarray(out["pts_mean"])[0]           # (S, K+1, 2)
        class_probs = np.asarray(jax.nn.softmax(out["class_logits"], axis=-1))[0]
        p_flag = class_probs[:, CLASS_FLAGELLUM]

        # Load raw source frame
        raw = load_raw_source(entry["source"])
        raw_disp = _norm(raw)
        rgb = np.stack([raw_disp] * 3, axis=-1)

        # Map GT native cropped-mask polylines back to raw pixel coords
        gt_native = entry["gt_polylines_native"]  # already in cropped-native space
        gt_raw = [native_cropped_to_full(g, entry["meta"]) for g in gt_native]

        # Map sample + mean predictions from canonical back to raw
        def to_raw(pts_canon):
            return native_cropped_to_full(canonical_pts_to_native_cropped(pts_canon, entry["meta"], cfg_c), entry["meta"])

        # Draw samples first (behind means)
        for s in range(pts_samples.shape[0]):
            a = float(min(1.0, p_flag[s]))
            if a < 0.05:
                continue
            color = SLOT_COLORS[s % len(SLOT_COLORS)]
            for n in range(min(pts_samples.shape[1], 20)):
                _draw_polyline(rgb, to_raw(pts_samples[s, n]), color, thickness=1,
                               alpha=args.sample_alpha * a)

        # Draw per-slot means on top
        for s in range(pts_mean.shape[0]):
            a = float(p_flag[s])
            if a < 0.1:
                continue
            color = SLOT_COLORS[s % len(SLOT_COLORS)]
            _draw_polyline(rgb, to_raw(pts_mean[s]), color, thickness=2, alpha=args.mean_alpha)

        # Draw GT green last (topmost, thick)
        for g in gt_raw:
            _draw_polyline(rgb, g, (0, 255, 0), thickness=2, alpha=1.0)

        # Annotate
        seq_short = entry["meta"]["sequence"].split("/")[-1][:35]
        cv2.putText(rgb, seq_short, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(rgb, f"p_flag max: {p_flag.max():.2f}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        Image.fromarray(rgb).save(Path(args.out_dir) / f"frame_{i:02d}_{entry['name']}")
        tiles.append(rgb)
        print(f"  frame {i}: {entry['meta']['sequence'][-40:]} raw={raw.shape} p_flag_max={p_flag.max():.2f}")

    print(f"[save] gallery: {args.out_dir}/")


if __name__ == "__main__":
    main()
