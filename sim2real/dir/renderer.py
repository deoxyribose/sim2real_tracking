"""Numpy renderer for a flagellum hypothesis — used both to build the
reconstruction cost and to visualize DIR solutions.

We rasterize a Gaussian tube along the skeleton polyline. Signed amp
convention matches the sim: amp>0 = brighter than BG (positive residual),
amp<0 = darker (negative residual).
"""
from __future__ import annotations

import numpy as np


def render_flagellum(skeleton_yx: np.ndarray, width_px: float, amp: float,
                     H: int, W: int, n_curve_samples: int = 64) -> np.ndarray:
    """Rasterize a (K, 2) skeleton as a Gaussian tube of the given width
    and signed amplitude. Returns (H, W) float32 signed."""
    if skeleton_yx.shape[0] < 2:
        return np.zeros((H, W), dtype=np.float32)

    # Densely sample points along the polyline
    ts = np.linspace(0.0, 1.0, n_curve_samples)
    K = skeleton_yx.shape[0]
    seg_ts = ts * (K - 1)
    idx_lo = np.clip(np.floor(seg_ts).astype(int), 0, K - 2)
    frac = seg_ts - idx_lo
    lo = skeleton_yx[idx_lo]
    hi = skeleton_yx[idx_lo + 1]
    curve = lo * (1 - frac[:, None]) + hi * frac[:, None]     # (n_samples, 2)

    sigma = max(width_px / 2.5, 0.5)
    r = int(np.ceil(3 * sigma))
    if r < 1:
        r = 1
    y_range = np.arange(-r, r + 1)
    x_range = np.arange(-r, r + 1)
    yy, xx = np.meshgrid(y_range, x_range, indexing="ij")
    kern = np.exp(-(yy ** 2 + xx ** 2) / (2 * sigma ** 2)).astype(np.float32)

    img = np.zeros((H, W), dtype=np.float32)
    for py, px in curve:
        iy, ix = int(round(py)), int(round(px))
        y0, y1 = max(0, iy - r), min(H, iy + r + 1)
        x0, x1 = max(0, ix - r), min(W, ix + r + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        ky0 = y0 - (iy - r)
        kx0 = x0 - (ix - r)
        ky1 = ky0 + (y1 - y0)
        kx1 = kx0 + (x1 - x0)
        img[y0:y1, x0:x1] = np.maximum(img[y0:y1, x0:x1], kern[ky0:ky1, kx0:kx1])
    return img * amp


def hypothesis_mask(skeleton_yx: np.ndarray, width_px: float,
                    H: int, W: int, n_curve_samples: int = 48) -> np.ndarray:
    """Boolean support mask of a hypothesis (union of nonzero pixels in a
    3-σ neighbourhood of the skeleton). Used for overlap tests."""
    r = int(np.ceil(1.5 * max(width_px, 1.0)))
    if skeleton_yx.shape[0] < 2:
        return np.zeros((H, W), dtype=bool)
    ts = np.linspace(0.0, 1.0, n_curve_samples)
    K = skeleton_yx.shape[0]
    seg_ts = ts * (K - 1)
    idx_lo = np.clip(np.floor(seg_ts).astype(int), 0, K - 2)
    frac = seg_ts - idx_lo
    lo = skeleton_yx[idx_lo]
    hi = skeleton_yx[idx_lo + 1]
    curve = lo * (1 - frac[:, None]) + hi * frac[:, None]

    mask = np.zeros((H, W), dtype=bool)
    for py, px in curve:
        iy, ix = int(round(py)), int(round(px))
        y0, y1 = max(0, iy - r), min(H, iy + r + 1)
        x0, x1 = max(0, ix - r), min(W, ix + r + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        mask[y0:y1, x0:x1] = True
    return mask
