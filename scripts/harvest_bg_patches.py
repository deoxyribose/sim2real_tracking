"""Harvest real background residual patches from canonicalized clips.

For each sequence:
  1. Sample several 16-frame clips at random positions.
  2. Canonicalize each clip.
  3. Compute temporal-energy map.
  4. Find sub-patch centers with LOW energy (certified flagella-free).
  5. Extract (T, patch_h, patch_w) clips and save.

The saved patches are used at training time to composite behind sim FG. Includes
debris, mottle, real noise — the parts of the sim-to-real gap our simulator
can't model.
"""
import argparse
import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from sim2real.data.canonicalize import CanonicalConfig, canonicalize_clip

CALIB_PATH = Path("/home/frans/sim2real_tracking/annotations/flagella_v0/calibration.json")
BASE = Path("/home/frans/sim2real_tracking/algae_data/CC124_pipette_wt_beat")

# Per-sequence banner-row crops we detected in annot_sample_frames.py. Keep minimal —
# if unknown, default to (0, orig_h) i.e. no crop.
DEFAULT_BANNER_CROP = {
    # sequence relative path → (crop_top, crop_bot)
    "1912/uni_rotor/cell1_20241219_154532": (26, 208),
    "1912/uni_rotor/cell_2_20241219_164118": (26, 268),
    "1912/uni_rotor/cell_3_4_20241219_171504": (26, 300),
    "1912/uni_rotor/cell_5_20241219_171909": (26, 288),
    "1912/uni_rotor/flagellum_1_20241219_155956": (26, 900),
    "CC124_cell1/2021_10_07/Export_20211007_174615": (2, 176),
    "CC124_cell1/2021_10_07/Export_20211007_180313": (2, 176),
    "CC124_cell2/Export_20211015_124200": (2, 200),
    "CC124_cell2/Export_20211015_124646": (2, 200),
    "CC124_cell2/Export_20211015_124944": (2, 200),
    "CC124_cell2/Export_20211015_125242": (2, 200),
}


SKIP_PATTERNS = ("_cropped", "_subtracted", "_block_mean")


def load_native_clip(seq_root: str, start: int, count: int, crop: tuple[int, int]) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(seq_root, "*.bmp")) + glob.glob(os.path.join(seq_root, "*.tif")))
    files = files[start : start + count]
    if not files:
        raise RuntimeError(f"no frames at {seq_root} start={start}")
    frames = []
    top, bot = crop
    for f in files:
        img = tifffile.imread(f) if f.endswith(".tif") else np.array(Image.open(f))
        if img.ndim == 3:
            img = img.mean(axis=-1)
        img = img[top:bot, :] if bot else img[top:, :]
        frames.append(img)
    return np.stack(frames).astype(np.float32)


def find_low_energy_centers(energy: np.ndarray, patch_h: int, patch_w: int, n: int,
                            pct_thresh: float = 20.0, min_center_dist: int = 20) -> list[tuple[int, int]]:
    """Return up to n (y, x) center coords whose surrounding patch has mean energy
    below the pct_thresh-percentile of the whole energy map.

    Enforces a minimum distance between chosen centers to encourage patch diversity.
    """
    H, W = energy.shape
    py, px = patch_h // 2, patch_w // 2
    # Valid centers: must fit the patch inside the map
    ys = np.arange(py, H - py, 4)
    xs = np.arange(px, W - px, 4)
    if len(ys) == 0 or len(xs) == 0:
        return []

    # Compute patch-mean energy for each candidate center via integral image
    ii = np.cumsum(np.cumsum(energy, axis=0), axis=1)
    def patch_mean(y0, x0, y1, x1):
        # (y0,x0) inclusive, (y1,x1) inclusive
        y0m, x0m = y0 - 1, x0 - 1
        A = ii[y1, x1]
        B = ii[y0m, x1] if y0m >= 0 else 0
        C = ii[y1, x0m] if x0m >= 0 else 0
        D = ii[y0m, x0m] if (y0m >= 0 and x0m >= 0) else 0
        area = (y1 - y0 + 1) * (x1 - x0 + 1)
        return (A - B - C + D) / area

    candidates = []
    for y in ys:
        for x in xs:
            m = patch_mean(y - py, x - px, y + py - 1, x + px - 1)
            candidates.append((float(m), int(y), int(x)))
    candidates.sort(key=lambda t: t[0])
    thresh = float(np.percentile([c[0] for c in candidates], pct_thresh))

    picked: list[tuple[int, int]] = []
    for m, y, x in candidates:
        if m > thresh:
            break
        if all((abs(y - yp) + abs(x - xp)) >= min_center_dist for yp, xp in picked):
            picked.append((y, x))
            if len(picked) >= n:
                break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/frans/sim2real_tracking/data_cache/bg_patches_v0.npz")
    ap.add_argument("--patch-h", type=int, default=96)
    ap.add_argument("--patch-w", type=int, default=96)
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--clips-per-seq", type=int, default=4)
    ap.add_argument("--patches-per-clip", type=int, default=6)
    ap.add_argument("--pct-thresh", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    calib = json.load(open(CALIB_PATH))["per_sequence"]

    all_patches = []       # each (T, patch_h, patch_w) float32
    all_sequences = []      # sequence name per patch (for provenance)

    # Enumerate sequences (skip derived variants)
    seqs = []
    for root, dirs, files in os.walk(BASE):
        imgs = sorted(f for f in files if f.endswith((".bmp", ".tif")))
        if not imgs:
            continue
        rel = os.path.relpath(root, BASE)
        if any(p in rel for p in SKIP_PATTERNS):
            continue
        seqs.append((rel, root, imgs))
    seqs.sort()
    print(f"[scan] {len(seqs)} sequences")

    for rel, root, imgs in seqs:
        n_frames = len(imgs)
        if rel in calib:
            src_w = calib[rel]["width_px"]["p50"]
        else:
            src_w = 4.0
            print(f"  {rel}: no calibration, using src_width_px={src_w}")
        crop = DEFAULT_BANNER_CROP.get(rel, (0, 0))
        cfg = CanonicalConfig(src_width_px=src_w, bg_median_window=15)
        n_this = 0
        for _ in range(args.clips_per_seq):
            start = rng.randint(0, max(0, n_frames - args.clip_len - 1))
            try:
                raw = load_native_clip(root, start, args.clip_len, crop)
            except Exception as e:
                print(f"  {rel}: load fail: {e}")
                continue
            out = canonicalize_clip(raw, cfg)
            clip = out["clip"]        # (T, H, W)
            energy = out["energy"]    # (H, W)
            centers = find_low_energy_centers(energy, args.patch_h, args.patch_w,
                                              n=args.patches_per_clip,
                                              pct_thresh=args.pct_thresh)
            for (y, x) in centers:
                py, px = args.patch_h // 2, args.patch_w // 2
                patch = clip[:, y - py : y + py, x - px : x + px].copy()
                if patch.shape == (args.clip_len, args.patch_h, args.patch_w):
                    all_patches.append(patch)
                    all_sequences.append(rel)
                    n_this += 1
        print(f"  {rel}: {n_this} patches (src_w={src_w:.2f}, crop={crop})")

    stacked = np.stack(all_patches, axis=0)  # (N, T, patch_h, patch_w)
    print(f"[save] {stacked.shape} float32 = {stacked.nbytes/1e6:.1f} MB → {args.out}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, patches=stacked.astype(np.float32),
                        sequences=np.array(all_sequences, dtype=object),
                        patch_h=args.patch_h, patch_w=args.patch_w, clip_len=args.clip_len)


if __name__ == "__main__":
    main()
