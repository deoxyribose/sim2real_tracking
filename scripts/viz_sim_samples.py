"""Render sim samples for visual inspection of the BG pipeline.

Two figures:
  before_after.png — same scene seed composited with the OLD bank + symmetric-pad tiling
                     vs the repaired bank + randomized flip-tiling.
  samples.png      — N sampled clips, each shown without and with its GT latents overlaid.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sim2real.data import CANONICAL_H, CANONICAL_W, FlagellumSimConfig, sample_scene
from sim2real.data.sim_flagella import _render_clip
from sim2real.data.sim_flagella import _tile_bg  # noqa: F401  (kept for parity with sim path)

GREEN, CYAN, YELLOW = (60, 230, 90), (80, 210, 235), (255, 205, 40)


def legacy_tile_bg(bg_patch, H, W, T):
    """The pre-fix `_tile_bg`: mirror the patch about the canvas centre."""
    bg_T, bh, bw = bg_patch.shape
    bg_patch = (np.tile(bg_patch, ((T + bg_T - 1) // bg_T, 1, 1))[:T] if bg_T < T else bg_patch[:T])
    if bh < H or bw < W:
        ph, pw = max(0, H - bh), max(0, W - bw)
        bg_patch = np.pad(bg_patch, ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)),
                          mode="symmetric")
    if bg_patch.shape[1] > H or bg_patch.shape[2] > W:
        y0 = (bg_patch.shape[1] - H) // 2
        x0 = (bg_patch.shape[2] - W) // 2
        bg_patch = bg_patch[:, y0:y0 + H, x0:x0 + W]
    return bg_patch.astype(np.float32)


def to_rgb(frame, lo=None, hi=None):
    if lo is None:
        lo, hi = np.percentile(frame, [1, 99])
    g = np.clip((frame - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.stack([g] * 3, -1)


def overlay_labels(rgb, latents):
    """Draw the sim's own sampled latents: cell disks, flagellum centrelines, attachments."""
    out = rgb.copy()
    for c in latents.cells:
        cy, cx = int(round(c.center[0])), int(round(c.center[1]))
        cv2.circle(out, (cx, cy), int(round(c.radius_px)), CYAN, 1, cv2.LINE_AA)
        cv2.drawMarker(out, (cx, cy), CYAN, cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
    for f in latents.flagella:
        pts = f.all_points()
        xy = np.stack([pts[:, 1], pts[:, 0]], 1).astype(np.int32)
        cv2.polylines(out, [xy], False, GREEN, 2, cv2.LINE_AA)
        ay, ax = int(round(f.attachment[0])), int(round(f.attachment[1]))
        cv2.circle(out, (ax, ay), 4, YELLOW, -1, cv2.LINE_AA)
    return out


def label_bar(w, text, h=20):
    bar = np.full((h, w, 3), 22, np.uint8)
    cv2.putText(bar, text, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
    return bar


def stack_titled(panels, titles):
    return np.concatenate([np.concatenate([label_bar(p.shape[1], t), p], 0)
                           for p, t in zip(panels, titles)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-old", default="data_cache/bg_patches_v0.npz")
    ap.add_argument("--bank-new", default="data_cache/bg_patches_v1.npz")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="experiments/detr_slot_v2/viz_sim_fixed")
    ap.add_argument("--crowded", action="store_true",
                    help="Widen the scene-count prior only (more cells, more flagella per "
                         "cell). Flagellum shape/beat/width/amplitude priors are untouched.")
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--min-flagella", type=int, default=0,
                    help="Rejection-sample scenes until at least this many flagella.")
    args = ap.parse_args()

    old_bank = np.load(args.bank_old, allow_pickle=True)["patches"]
    new_bank = np.load(args.bank_new, allow_pickle=True)["patches"]
    if args.crowded:
        cfg = FlagellumSimConfig(T=16, p_empty_scene=0.0,
                                 n_cells_probs=(0.0, 0.15, 0.35, 0.35, 0.15),
                                 n_flagella_per_cell_probs=(0.0, 0.15, 0.45, 0.40))
    else:
        cfg = FlagellumSimConfig(T=16)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mid = cfg.T // 2

    # ---------------- before / after ----------------
    rows = []
    if args.crowded:
        rows = None
    for i in range(0 if args.crowded else args.n):
        # Identical foreground: same seed for the scene, only the BG path differs.
        scene = sample_scene(np.random.default_rng(args.seed + i), cfg, bg_patch=None)
        fg = scene.clip
        ro = np.random.default_rng(1000 + i)
        old_bg = legacy_tile_bg(old_bank[ro.integers(len(old_bank))], CANONICAL_H, CANONICAL_W, cfg.T)
        new_bg = _tile_bg(new_bank[ro.integers(len(new_bank))], CANONICAL_H, CANONICAL_W, cfg.T,
                          rng=np.random.default_rng(2000 + i))
        rows.append(stack_titled(
            [to_rgb((fg + old_bg)[mid]), to_rgb((fg + new_bg)[mid])],
            [f"[{i}] BEFORE  v0 bank + symmetric pad", f"[{i}] AFTER  v1 bank + flip-tiling"]))
    if rows:
        Image.fromarray(np.concatenate(rows, 0)).save(out_dir / "before_after.png")

    # ---------------- with / without labels ----------------
    rows = []
    rng = np.random.default_rng(args.seed)
    for i in range(args.n):
        for _ in range(500):
            bg = new_bank[rng.integers(len(new_bank))]
            scene = sample_scene(rng, cfg, bg_patch=bg)
            if len(scene.latents.flagella) >= args.min_flagella:
                break
        frame = to_rgb(scene.clip[mid])
        n_f, n_c = len(scene.latents.flagella), len(scene.latents.cells)
        rows.append(stack_titled(
            [frame, overlay_labels(frame, scene.latents)],
            [f"[{i}] no labels", f"[{i}] labels: {n_c} cell(s), {n_f} flagell(a)"]))
    name = args.out_name or ("samples_crowded.png" if args.crowded else "samples.png")
    Image.fromarray(np.concatenate(rows, 0)).save(out_dir / name)
    print(f"[save] {out_dir}/{name}")


if __name__ == "__main__":
    main()
