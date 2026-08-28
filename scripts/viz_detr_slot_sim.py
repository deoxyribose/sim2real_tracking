"""Visualize DETR-slot predictions on SIM data (the training distribution).

Since sim gives us the exact ground-truth latents, we can compare model output
directly to what the sim generated. This is the "in-distribution" check —
performance here upper-bounds real-frame performance.
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

from sim2real.data import (
    CANONICAL_H, CANONICAL_W, CANONICAL_TARGET_WIDTH_PX, CLASS_FLAGELLUM,
    FlagellumSimConfig, sample_scene,
)
from sim2real.model_v2 import DETRSlotModel, sample_flagellum_from_head


SLOT_COLORS = [
    (255, 60, 60), (60, 200, 60), (60, 120, 255), (255, 200, 30),
    (200, 60, 255), (30, 220, 220), (255, 120, 60), (150, 220, 60),
]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/frans/sim2real_tracking/runs/detr_slot/v1/ckpt_step_20000.pkl")
    ap.add_argument("--bg-patches", default="/home/frans/sim2real_tracking/data_cache/bg_patches_v0.npz")
    ap.add_argument("--n-frames", type=int, default=12)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--sample-alpha", type=float, default=0.5)
    ap.add_argument("--mean-alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/home/frans/sim2real_tracking/runs/detr_slot/v1/viz_sim")
    args = ap.parse_args()

    ckpt = pickle.load(open(args.ckpt, "rb"))
    model = DETRSlotModel(cfg=ckpt["model_cfg"])
    bg_patches = np.load(args.bg_patches, allow_pickle=True)["patches"]
    sim_cfg = FlagellumSimConfig(T=16)

    rng = np.random.default_rng(args.seed)

    @jax.jit
    def fwd(p, c, e, k):
        return model.apply(p, c, e, rngs={"slots": k})

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(args.seed)

    for i in range(args.n_frames):
        bg = bg_patches[rng.integers(0, len(bg_patches))]
        scene = sample_scene(rng, sim_cfg, bg_patch=bg)
        clip = jnp.asarray(scene.clip)[None]
        energy = jnp.asarray(scene.energy)[None]
        key, sub = jax.random.split(key)
        out = fwd(ckpt["params"], clip, energy, sub)
        key, sub2 = jax.random.split(key)
        samples = sample_flagellum_from_head(sub2, out, n_samples=args.n_samples, temperature=1.0)
        pts_samples = np.asarray(samples["pts_samples"])[0]
        pts_mean = np.asarray(out["pts_mean"])[0]
        class_probs = np.asarray(jax.nn.softmax(out["class_logits"], axis=-1))[0]
        p_flag = class_probs[:, CLASS_FLAGELLUM]

        # Base frame = sim clip midpoint
        mid = scene.clip[scene.clip.shape[0] // 2]
        rgb = np.stack([_norm(mid)] * 3, axis=-1)

        # Samples
        for s in range(pts_samples.shape[0]):
            a = float(min(1.0, p_flag[s]))
            if a < 0.05:
                continue
            color = SLOT_COLORS[s % len(SLOT_COLORS)]
            for n in range(min(pts_samples.shape[1], 20)):
                _draw_polyline(rgb, pts_samples[s, n], color, thickness=1,
                               alpha=args.sample_alpha * a)

        # Per-slot means
        for s in range(pts_mean.shape[0]):
            a = float(p_flag[s])
            if a < 0.1:
                continue
            color = SLOT_COLORS[s % len(SLOT_COLORS)]
            _draw_polyline(rgb, pts_mean[s], color, thickness=2, alpha=args.mean_alpha)

        # GT flagella (green)
        for latent in scene.latents.flagella:
            _draw_polyline(rgb, latent.all_points(), (0, 255, 0), thickness=2, alpha=1.0)
            # attachment as bright green dot
            ay, ax = int(round(latent.attachment[0])), int(round(latent.attachment[1]))
            cv2.circle(rgb, (ax, ay), 4, (0, 255, 60), -1)

        # Annotate
        n_flag = len(scene.latents.flagella)
        cv2.putText(rgb, f"GT flagella: {n_flag}   p_flag max: {p_flag.max():.2f}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        Image.fromarray(rgb).save(Path(args.out_dir) / f"sim_{i:02d}.png")
        print(f"  sim {i}: GT={n_flag} flagella, p_flag_max={p_flag.max():.2f}")

    print(f"[save] gallery: {args.out_dir}/")


if __name__ == "__main__":
    main()
