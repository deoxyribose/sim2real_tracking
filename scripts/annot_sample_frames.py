"""Sample frames from all algae sequences for annotation.

For each sequence:
  1. Enumerate frames (TIF or BMP).
  2. Auto-detect metadata banner rows via row-wise std/mean heuristic.
  3. Sample N frames evenly across the sequence.
  4. Crop banners, contrast-stretch, resize (max side = canvas), pad to canvas x canvas.
  5. Save as PNG in one flat directory: img_000.png ... img_NNN.png
  6. Write manifest.json with the source path + crop/resize/pad transform for each.

Usage:
  python scripts/annot_sample_frames.py --n-per-seq 4 --canvas 512 --out ...
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

DEFAULT_BASE = "/home/frans/sim2real_tracking/algae_data/CC124_pipette_wt_beat"

# Explicitly skip derivative folders (cropped/subtracted variants of a raw sequence).
SKIP_PATTERNS = ("_cropped", "_subtracted", "_block_mean")


def find_sequences(base):
    seqs = []
    for root, _dirs, files in os.walk(base):
        imgs = sorted(f for f in files if f.endswith((".bmp", ".tif")))
        if not imgs:
            continue
        rel = os.path.relpath(root, base)
        if any(p in rel for p in SKIP_PATTERNS):
            continue
        seqs.append((rel, root, imgs))
    return sorted(seqs)


def load_frame(path):
    if path.endswith(".tif"):
        return tifffile.imread(path)
    return np.array(Image.open(path))


def detect_banner_rows(img: np.ndarray) -> tuple[int, int]:
    """Return (top_crop, bottom_crop) rows that are pure image content.

    Banner rows have anomalous stats: either pure white (std<4, mean>240) or text-noisy
    (std>40 with mean<220). We scan from each edge inward until we hit a run of >=5
    consecutive clean rows.
    """
    if img.ndim == 3:
        img = img.mean(axis=-1)
    h = img.shape[0]
    rs, rm = img.std(axis=1), img.mean(axis=1)

    def is_banner(i):
        s, m = rs[i], rm[i]
        return (s < 4 and m > 240) or (s > 40 and m < 220)

    top = 0
    for i in range(h):
        if not any(is_banner(i + k) for k in range(min(5, h - i))):
            top = i
            break
    bot = h
    for i in range(h - 1, -1, -1):
        if not any(is_banner(i - k) for k in range(min(5, i + 1))):
            bot = i + 1
            break
    if bot - top < 20:  # sanity — never crop away the whole image
        return 0, h
    return top, bot


def contrast_stretch(gray: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(gray, [lo_pct, hi_pct])
    if hi <= lo:
        return gray.astype(np.uint8)
    out = np.clip((gray.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255)
    return out.astype(np.uint8)


def fit_to_canvas(gray: np.ndarray, canvas: int) -> tuple[np.ndarray, dict]:
    """Resize (keep aspect) then pad to canvas x canvas. Return image + inverse transform info."""
    h, w = gray.shape
    scale = canvas / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = np.array(Image.fromarray(gray).resize((new_w, new_h), Image.LANCZOS))
    pad_top = (canvas - new_h) // 2
    pad_left = (canvas - new_w) // 2
    fill = int(resized.mean())
    out = np.full((canvas, canvas), fill, dtype=np.uint8)
    out[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    tform = dict(scale=float(scale), pad_top=int(pad_top), pad_left=int(pad_left),
                 orig_h=int(h), orig_w=int(w))
    return out, tform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--n-per-seq", type=int, default=4)
    ap.add_argument("--canvas", type=int, default=512)
    ap.add_argument("--out", default="/home/frans/sim2real_tracking/annotations/flagella_v0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames_dir = Path(args.out) / "frames"
    masks_dir = Path(args.out) / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    seqs = find_sequences(args.base)
    print(f"[scan] {len(seqs)} sequences found under {args.base}")
    rng = np.random.default_rng(args.seed)

    manifest = []
    idx = 0
    for rel, root, imgs in seqs:
        n = len(imgs)
        # Even sampling with small jitter so we don't always pick the same phase of the beat.
        anchors = np.linspace(0, n - 1, args.n_per_seq + 2, dtype=int)[1:-1]
        jitter = rng.integers(-20, 21, size=len(anchors))
        picks = np.clip(anchors + jitter, 0, n - 1)
        picks = np.unique(picks).tolist()

        # Detect banner rows using the first sampled frame; assume constant per sequence.
        sample = load_frame(os.path.join(root, imgs[picks[0]]))
        top, bot = detect_banner_rows(sample)

        for frame_idx in picks:
            src_path = os.path.join(root, imgs[frame_idx])
            raw = load_frame(src_path)
            gray = raw if raw.ndim == 2 else raw.mean(axis=-1).astype(raw.dtype)
            cropped = gray[top:bot, :]
            stretched = contrast_stretch(cropped)
            canvas_img, tform = fit_to_canvas(stretched, args.canvas)

            out_name = f"img_{idx:03d}.png"
            Image.fromarray(canvas_img).save(frames_dir / out_name)

            manifest.append(dict(
                idx=idx,
                out_name=out_name,
                source=src_path,
                sequence=rel,
                frame_in_seq=int(frame_idx),
                crop_top=int(top),
                crop_bot=int(bot),
                canvas=args.canvas,
                **tform,
            ))
            idx += 1

    with open(Path(args.out) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[save] {idx} frames -> {frames_dir}")
    print(f"[save] manifest -> {Path(args.out) / 'manifest.json'}")
    print(f"[next] launch the annotator: python scripts/annot_flagella.py --dir {args.out}")


if __name__ == "__main__":
    main()
