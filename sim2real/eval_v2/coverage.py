"""Sample-coverage-recall metric on the 59 real annotated frames.

For each annotated frame:
  1. Load the source video, canonicalize a clip centered on the annotated frame.
  2. Run the model on the canonicalized clip.
  3. Sample N candidate curves per slot from the predicted Gaussians.
  4. For each GT flagellum, compute the min Chamfer distance to ANY sampled candidate curve.
  5. GT is "covered" if that min distance ≤ k * canonical_width (k configurable, default 2).

Metric = fraction of GT annotations covered across all frames.

This is the model-selection metric — not NLL. Coverage tells us whether the model's
DISTRIBUTIONS contain the right answer; that's what the downstream verifier + CP-SAT
needs, not a precise point estimate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import binary_dilation, label as cc_label
from skimage.morphology import skeletonize

from ..data import (
    CANONICAL_H, CANONICAL_W, CANONICAL_TARGET_WIDTH_PX,
    CanonicalConfig, canonicalize_clip,
)


ANNOT_ROOT = Path("/home/frans/sim2real_tracking/annotations/flagella_v0")


def canvas_mask_to_cropped(mask_canvas: np.ndarray, meta: dict) -> np.ndarray:
    """Undo annot_sample_frames.py transform, back to cropped-native space."""
    pad_top, pad_left = meta["pad_top"], meta["pad_left"]
    orig_h, orig_w = meta["orig_h"], meta["orig_w"]
    scale = meta["scale"]
    H_pad = int(round(orig_h * scale))
    W_pad = int(round(orig_w * scale))
    unpadded = mask_canvas[pad_top : pad_top + H_pad, pad_left : pad_left + W_pad]
    if unpadded.shape != (orig_h, orig_w):
        im = Image.fromarray((unpadded > 127).astype(np.uint8) * 255)
        im = im.resize((orig_w, orig_h), Image.NEAREST)
        unpadded = np.array(im)
    return unpadded > 127


def gt_polylines_from_mask(cropped_mask: np.ndarray, min_len: int = 6) -> list[np.ndarray]:
    """Extract per-CC skeletons as ordered polylines (yx coords in cropped-native space)."""
    if not cropped_mask.any():
        return []
    skel = skeletonize(cropped_mask)
    labeled, n = cc_label(cropped_mask)
    polylines = []
    for k in range(1, n + 1):
        cc = labeled == k
        skel_k = skel & cc
        if skel_k.sum() < min_len:
            continue
        # Order skeleton pixels starting from an endpoint (skeleton pixel with exactly 1 neighbor).
        pts = _order_skeleton(skel_k)
        if len(pts) >= 2:
            polylines.append(pts.astype(np.float32))
    return polylines


def _order_skeleton(skel: np.ndarray) -> np.ndarray:
    """Greedy walk from an endpoint through skeleton pixels. Returns (N, 2) in (y, x) order."""
    from scipy.ndimage import convolve
    kern = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    nc = convolve(skel.astype(np.uint8), kern, mode="constant", cval=0)
    endpoints = np.stack(np.where(skel & (nc == 1)), axis=1)
    if len(endpoints) == 0:
        # closed loop or too short — return raw pixel list
        return np.stack(np.where(skel), axis=1)
    start = endpoints[0]
    visited = set()
    path = [tuple(start)]
    visited.add(tuple(start))
    while True:
        y, x = path[-1]
        found = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1] and skel[ny, nx] and (ny, nx) not in visited:
                    path.append((ny, nx))
                    visited.add((ny, nx))
                    found = True
                    break
            if found:
                break
        if not found:
            break
    return np.array(path, dtype=np.int32)


def gt_polyline_to_canonical(polyline: np.ndarray, meta: dict, cfg: CanonicalConfig,
                              canonical_h: int = CANONICAL_H, canonical_w: int = CANONICAL_W,
                              cropped_h: Optional[int] = None, cropped_w: Optional[int] = None) -> np.ndarray:
    """Transform a cropped-native polyline (N, 2) yx into canonical (yc, xc) coords.

    Steps mirror canonicalize_clip: resample_scale, then center-crop-or-pad to canonical size.
    We use the source's cropped dims (meta.orig_h/orig_w = cropped size).
    """
    if cropped_h is None:
        cropped_h = meta["orig_h"]
    if cropped_w is None:
        cropped_w = meta["orig_w"]
    scale = cfg.resample_scale
    # Post-resample dims
    new_h = int(round(cropped_h * scale))
    new_w = int(round(cropped_w * scale))
    pts = polyline.astype(np.float32) * scale
    # Center-crop or pad to canonical
    pad_top = (canonical_h - new_h) // 2
    pad_left = (canonical_w - new_w) // 2
    pts[:, 0] += pad_top
    pts[:, 1] += pad_left
    return pts


def load_source_cropped(src_path: str, meta: dict) -> np.ndarray:
    """Load source frame and apply the same banner crop the sampler used."""
    img = tifffile.imread(src_path) if src_path.endswith(".tif") else np.array(Image.open(src_path))
    if img.ndim == 3:
        img = img.mean(axis=-1)
    return img[meta["crop_top"] : meta["crop_bot"], :]


def load_source_clip(src_path: str, meta: dict, T: int = 16) -> np.ndarray:
    """Load a T-frame clip centered on the annotated frame, apply banner crop."""
    import glob
    import os
    dir_ = os.path.dirname(src_path)
    ext = "." + src_path.rsplit(".", 1)[1]
    all_files = sorted(glob.glob(os.path.join(dir_, "*" + ext)))
    center_idx = all_files.index(src_path)
    half = T // 2
    start = max(0, center_idx - half)
    end = min(len(all_files), start + T)
    start = max(0, end - T)
    files = all_files[start:end]
    frames = []
    for f in files:
        img = tifffile.imread(f) if f.endswith(".tif") else np.array(Image.open(f))
        if img.ndim == 3:
            img = img.mean(axis=-1)
        frames.append(img[meta["crop_top"] : meta["crop_bot"], :])
    # Pad in time if we hit an edge
    while len(frames) < T:
        frames.append(frames[-1])
    return np.stack(frames).astype(np.float32)


def canonicalize_real_frame(annotation_meta: dict, src_width_px: float, T: int = 16) -> dict:
    """Load clip around the annotated frame, canonicalize, return canonical clip + energy."""
    src_clip = load_source_clip(annotation_meta["source"], annotation_meta, T=T)
    cfg = CanonicalConfig(src_width_px=src_width_px, bg_median_window=min(15, T))
    return canonicalize_clip(src_clip, cfg), cfg


def load_real_annotations(calibration_path: Path = ANNOT_ROOT / "calibration.json",
                           status_path: Path = ANNOT_ROOT / "status.json",
                           manifest_path: Path = ANNOT_ROOT / "manifest.json") -> list[dict]:
    """Return list of {name, meta, source, src_width_px, gt_polylines_native}."""
    manifest = json.load(open(manifest_path))
    status = json.load(open(status_path))
    calib = json.load(open(calibration_path))
    per_seq = calib["per_sequence"]
    out = []
    for m in manifest:
        if status.get(m["out_name"]) != "annotated":
            continue
        seq = m["sequence"]
        src_w = per_seq[seq]["width_px"]["p50"] if seq in per_seq else 4.0
        canvas_mask = np.array(Image.open(ANNOT_ROOT / "masks" / m["out_name"].replace("img_", "mask_")))
        cropped_mask = canvas_mask_to_cropped(canvas_mask, m)
        gt_polylines_native = gt_polylines_from_mask(cropped_mask)
        if not gt_polylines_native:
            continue
        out.append(dict(name=m["out_name"], meta=m, source=m["source"],
                        src_width_px=src_w, gt_polylines_native=gt_polylines_native))
    return out


# ------------------------------- Metric --------------------------------

def _chamfer_polylines(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    """Symmetric Chamfer distance between two polyline point clouds."""
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.inf
    d = np.linalg.norm(pred_pts[:, None] - gt_pts[None], axis=-1)  # (P, G)
    return 0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean())


def sample_coverage_recall(pred_samples: np.ndarray, gt_polylines_canonical: list[np.ndarray],
                            coverage_k: float = 2.0,
                            canonical_width_px: float = CANONICAL_TARGET_WIDTH_PX) -> dict:
    """Compute coverage recall for one frame.

    pred_samples: (S * n_samples, K+1, 2) — all sampled candidate polylines from all slots
    gt_polylines_canonical: list of (Ng, 2) polylines in canonical (y, x) space
    coverage_k: a GT polyline is 'covered' if some sample is within k * canonical_width in Chamfer.

    Returns dict with:
      covered:      (n_gt,) bool
      min_chamfer:  (n_gt,) float — distance to closest sample per GT
      recall:       float — fraction of covered
    """
    thresh = coverage_k * canonical_width_px
    if len(gt_polylines_canonical) == 0:
        return dict(covered=np.array([], dtype=bool), min_chamfer=np.array([]), recall=1.0)
    covered = []
    min_dists = []
    for gt in gt_polylines_canonical:
        best = np.inf
        for k in range(len(pred_samples)):
            d = _chamfer_polylines(pred_samples[k], gt)
            if d < best:
                best = d
                if best <= thresh:
                    break
        min_dists.append(best)
        covered.append(best <= thresh)
    recall = float(sum(covered) / max(len(covered), 1))
    return dict(covered=np.array(covered), min_chamfer=np.array(min_dists), recall=recall)
