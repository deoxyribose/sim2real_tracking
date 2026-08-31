"""Render sim clips as labelled videos.

Each frame shows:
  cyan   — cell disk + centre
  yellow — flagellum attachment on the membrane
  green  — the STORED GT rest shape (static; this is what supervision sees)
  orange — the TRUE per-frame centreline after the beat displacement (sim-internal,
           not part of the latent)

The gap between green and orange is the beat, which the labels do not capture.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from sim2real.data import FlagellumSimConfig, sample_scene
from sim2real.data.sim_flagella import _apply_beat

CYAN, YELLOW, GREEN, ORANGE = (80, 210, 235), (255, 205, 40), (60, 230, 90), (255, 140, 40)


def to_rgb(frame, lo, hi):
    g = np.clip((frame - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.stack([g] * 3, -1)


def poly(img, pts, color, thickness=2):
    xy = np.stack([pts[:, 1], pts[:, 0]], 1).astype(np.int32)
    cv2.polylines(img, [xy], False, color, thickness, cv2.LINE_AA)


def poly_dashed(img, pts, color, thickness=2, dash=6, gap=4):
    """Dashed polyline, resampled by arc length so dashes are evenly spaced."""
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = cum[-1]
    if total < 1e-6:
        return
    n = max(int(total), 2)
    t = np.linspace(0, total, n)
    ys = np.interp(t, cum, pts[:, 0])
    xs = np.interp(t, cum, pts[:, 1])
    i = 0
    while i < n - 1:
        j = min(i + dash, n - 1)
        seg = np.stack([ys[i:j + 1], xs[i:j + 1]], 1)
        if len(seg) > 1:
            poly(img, seg, color, thickness)
        i = j + gap


def draw(frame_rgb, latents, beats, t_norm, show_labels=True):
    img = frame_rgb.copy()
    if not show_labels:
        return img
    for c in latents.cells:
        cy, cx = int(round(c.center[0])), int(round(c.center[1]))
        cv2.circle(img, (cx, cy), int(round(c.radius_px)), CYAN, 1, cv2.LINE_AA)
        cv2.drawMarker(img, (cx, cy), CYAN, cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
    for f, params in zip(latents.flagella, beats):
        b = params["beat"]
        cp_t = _apply_beat(f.control_points, f.attachment, b["base_angle"], t_norm,
                           b["lat_amp"], b["wave"], b["freq"], b["phase"])
        poly(img, np.concatenate([f.attachment[None], cp_t], 0), ORANGE, 2)   # true, beating
        poly_dashed(img, f.all_points(), GREEN, 2)                            # stored GT, static
        ay, ax = int(round(f.attachment[0])), int(round(f.attachment[1]))
        cv2.circle(img, (ax, ay), 4, YELLOW, -1, cv2.LINE_AA)
    return img


def banner(w, text, h=18):
    bar = np.full((h, w, 3), 22, np.uint8)
    cv2.putText(bar, text, (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1, cv2.LINE_AA)
    return bar


LEGEND = [(CYAN, "cell"), (YELLOW, "attachment"),
          (ORANGE, "true centreline (beating)"), (GREEN, "GT label (static rest shape)")]


def legend_strip(w, h=22):
    """Legend sized from real text metrics, with the font scaled down if it would overflow."""
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, 1
    swatch, pad, gap = 18, 6, 18
    for scale in (0.36, 0.32, 0.28, 0.24):
        widths = [cv2.getTextSize(t, font, scale, thick)[0][0] for _, t in LEGEND]
        total = 8 + sum(swatch + pad + wd + gap for wd in widths)
        if total <= w:
            break
    bar = np.full((h, w, 3), 22, np.uint8)
    x = 8
    for (color, label), wd in zip(LEGEND, widths):
        cv2.line(bar, (x, h // 2), (x + swatch, h // 2), color, 3, cv2.LINE_AA)
        x += swatch + pad
        cv2.putText(bar, label, (x, h // 2 + 4), font, scale, (225, 225, 225), thick, cv2.LINE_AA)
        x += wd + gap
    return bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="data_cache/bg_patches_v1.npz")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--min-flagella", type=int, default=2)
    ap.add_argument("--upscale", type=int, default=2)
    ap.add_argument("--out-dir", default="experiments/detr_slot_v2/viz_sim_videos")
    args = ap.parse_args()

    bank = np.load(args.bank, allow_pickle=True)["patches"]
    cfg = FlagellumSimConfig(T=args.T)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for i in range(args.n):
        for _ in range(500):
            bg = bank[rng.integers(len(bank))]
            scene, beats = sample_scene(rng, cfg, bg_patch=bg, return_beats=True)
            if len(scene.latents.flagella) >= args.min_flagella:
                break
        lo, hi = np.percentile(scene.clip, [1, 99])
        n_c, n_f = len(scene.latents.cells), len(scene.latents.flagella)

        frames = []
        for t in range(args.T):
            base = to_rgb(scene.clip[t], lo, hi)
            t_norm = t / max(args.T - 1, 1)
            left = draw(base, scene.latents, beats, t_norm, show_labels=False)
            right = draw(base, scene.latents, beats, t_norm, show_labels=True)
            pair = np.concatenate(
                [np.concatenate([banner(left.shape[1], f"raw    t={t:02d}/{args.T}"), left], 0),
                 np.concatenate([banner(right.shape[1],
                                        f"labels    {n_c} cell(s), {n_f} flagell(a)"), right], 0)], 1)
            pair = np.concatenate([pair, legend_strip(pair.shape[1])], 0)
            if args.upscale > 1:
                pair = cv2.resize(pair, None, fx=args.upscale, fy=args.upscale,
                                  interpolation=cv2.INTER_NEAREST)
            frames.append(pair)

        path = out_dir / f"sim_{i:02d}.mp4"
        imageio.mimwrite(path, frames, fps=args.fps, codec="libx264",
                         macro_block_size=1, quality=8)
        print(f"[save] {path}  ({n_c} cells, {n_f} flagella, {args.T} frames @ {args.fps}fps)")


if __name__ == "__main__":
    main()
