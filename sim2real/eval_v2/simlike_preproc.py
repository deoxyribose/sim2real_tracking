"""Lightweight real preprocessing that mimics sim's clip_median.

Real's canonicalize_clip does bandpass + sigma-scale → std ~1.6, which is
~200x larger than sim's clip_median (std ~0.007). This preproc uses only
median-subtract + /255, matching sim's raw range.
"""
from __future__ import annotations
import cv2, numpy as np

from sim2real.eval_v2.coverage import load_source_clip


def simlike_canonicalize(annotation_meta: dict, T: int, target_hw: tuple[int, int]):
    """Return (clip, static_median) in sim's clip_median / temporal_median units.

      clip: (T, target_h, target_w) float32, median-subtracted, /255
      static_median: (target_h, target_w) float32 in [0, 1]

    Also returns bookkeeping needed to project GT polylines to the target frame:
      src_h, src_w: source (banner-cropped) size
      sy, sx: y/x scale from source → target
    """
    src_clip = load_source_clip(annotation_meta["source"], annotation_meta, T=T)
    src_clip = src_clip.astype(np.float32) / 255.0
    src_h, src_w = src_clip.shape[1], src_clip.shape[2]
    target_h, target_w = target_hw
    sy = target_h / src_h; sx = target_w / src_w

    # Median-subtract in ORIGINAL resolution first (matches sim's convention:
    # per-frame residual against the temporal median).
    med = np.median(src_clip, axis=0)                       # (src_h, src_w)
    residual = src_clip - med                                # (T, src_h, src_w)

    # Resize to target
    clip_r = np.stack([cv2.resize(residual[t], (target_w, target_h),
                                     interpolation=cv2.INTER_AREA)
                         for t in range(residual.shape[0])], axis=0).astype(np.float32)
    smed_r = cv2.resize(med, (target_w, target_h),
                         interpolation=cv2.INTER_AREA).astype(np.float32)
    return clip_r, smed_r, src_h, src_w, sy, sx
