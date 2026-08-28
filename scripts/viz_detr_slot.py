"""Visualize DETR-slot predictions on real annotated frames.

For each frame: draw the canonicalized clip's midpoint frame, overlay the GT
polylines (green) and up to N sampled candidate curves per slot (thin red lines
whose brightness = probability of the sampling slot being flagellum). Also show
each slot's mean curve in a distinct color.

Emits a mosaic PNG plus a per-frame gallery.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from sim2real.data import CANONICAL_H, CANONICAL_W, CANONICAL_TARGET_WIDTH_PX, CLASS_FLAGELLUM, FLAGELLUM_K
from sim2real.eval_v2.coverage import (
    canonicalize_real_frame, gt_polyline_to_canonical, load_real_annotations,
)
from sim2real.model_v2 import DETRSlotModel, sample_flagellum_from_head


SLOT_COLORS = [
    (255, 60, 60), (60, 200, 60), (60, 120, 255), (255, 200, 30),
    (200, 60, 255), (30, 220, 220), (255, 120, 60), (150, 220, 60),
]


def _draw_polyline(img_rgb: np.ndarray, pts: np.ndarray, color, thickness: int = 1, alpha: float = 1.0):
    """In-place draw a polyline (yx format) with alpha blend."""
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
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--n-frames", type=int, default=16, help="Total frames to visualize")
    ap.add_argument("--out", default="/home/frans/sim2real_tracking/runs/detr_slot/v1/viz_samples.png")
    ap.add_argument("--gallery-dir", default="/home/frans/sim2real_tracking/runs/detr_slot/v1/viz_frames")
    args = ap.parse_args()

    ckpt = pickle.load(open(args.ckpt, "rb"))
    model = DETRSlotModel(cfg=ckpt["model_cfg"])
    ann = load_real_annotations()
    # Pick spread across sequences
    seen = {}
    picks = []
    for e in ann:
        s = e["meta"]["sequence"]
        if seen.get(s, 0) < 2:  # up to 2 frames per sequence
            picks.append(e)
            seen[s] = seen.get(s, 0) + 1
        if len(picks) >= args.n_frames:
            break
    print(f"[viz] {len(picks)} frames to render")

    @jax.jit
    def fwd(p, c, e, k):
        return model.apply(p, c, e, rngs={"slots": k})

    Path(args.gallery_dir).mkdir(parents=True, exist_ok=True)
    tiles = []
    key = jax.random.PRNGKey(0)
    for i, entry in enumerate(picks):
        canon, cfg_c = canonicalize_real_frame(entry["meta"], entry["src_width_px"], T=16)
        clip = jnp.asarray(canon["clip"])[None]
        energy = jnp.asarray(canon["energy"])[None]
        key, sub = jax.random.split(key)
        out = fwd(ckpt["params"], clip, energy, sub)
        key, sub2 = jax.random.split(key)
        samples = sample_flagellum_from_head(sub2, out, n_samples=args.n_samples, temperature=args.temperature)
        pts_samples = np.asarray(samples["pts_samples"])[0]  # (S, N, K+1, 2)
        pts_mean = np.asarray(out["pts_mean"])[0]           # (S, K+1, 2)
        class_probs = np.asarray(jax.nn.softmax(out["class_logits"], axis=-1))[0]  # (S, 4)
        p_flag = class_probs[:, CLASS_FLAGELLUM]

        # Base frame = canonicalized clip midpoint
        mid = canon["clip"][canon["clip"].shape[0] // 2]
        base = _norm(mid)
        rgb = np.stack([base] * 3, axis=-1)

        # Draw sampled curves (thin, alpha = p_flag / 3)
        for s in range(pts_samples.shape[0]):
            color = SLOT_COLORS[s % len(SLOT_COLORS)]
            alpha = float(min(1.0, p_flag[s]))
            if alpha < 0.05:
                continue
            for n in range(min(pts_samples.shape[1], 20)):  # cap for readability
                _draw_polyline(rgb, pts_samples[s, n], color, thickness=1, alpha=0.15 * alpha)

        # Draw per-slot MEAN in bold (alpha ∝ p_flag)
        for s in range(pts_mean.shape[0]):
            alpha = float(p_flag[s])
            if alpha < 0.1:
                continue
            color = SLOT_COLORS[s % len(SLOT_COLORS)]
            _draw_polyline(rgb, pts_mean[s], color, thickness=2, alpha=min(1.0, alpha * 1.5))

        # Draw GT polylines in bright green (thicker)
        gt_canonical = [gt_polyline_to_canonical(p, entry["meta"], cfg_c) for p in entry["gt_polylines_native"]]
        for g in gt_canonical:
            _draw_polyline(rgb, g, (0, 255, 0), thickness=2, alpha=1.0)

        # Annotate
        seq_short = entry["meta"]["sequence"].split("/")[-1][:25]
        cv2.putText(rgb, seq_short, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(rgb, f"p_flag max: {p_flag.max():.2f}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        tiles.append(rgb)
        Image.fromarray(rgb).save(Path(args.gallery_dir) / f"frame_{i:02d}_{entry['name']}")
        print(f"  frame {i}: {entry['meta']['sequence'][-40:]} (p_flag max={p_flag.max():.2f})")

    # Build mosaic
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    pad = 4
    H, W = tiles[0].shape[:2]
    canvas = np.full((rows * H + (rows + 1) * pad, cols * W + (cols + 1) * pad, 3), 40, dtype=np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        y = pad + r * (H + pad)
        x = pad + c * (W + pad)
        canvas[y:y+H, x:x+W] = t
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(args.out)
    print(f"[save] mosaic: {args.out}")
    print(f"[save] per-frame gallery: {args.gallery_dir}/")


if __name__ == "__main__":
    main()
