"""Canonicalization: real video / sim composite → canonical residual clip.

Pipeline
========
1. Running-median background subtract (over T)     → residual (T, H, W)
2. Band-pass at flagellum-width band (DoG)         → residual (T, H, W)
3. Width-normalize (resample so median flagellum   → residual (T, H', W')
   width becomes CANONICAL_TARGET_WIDTH_PX)
4. Center-crop / pad to canonical (H_c, W_c)       → residual (T, H_c, W_c)
5. σ-scale: divide by local noise std              → residual (T, H_c, W_c)
6. Temporal-energy map (variance over T)           → energy (H_c, W_c)

Output: (T, H_c, W_c) canonical residual + (H_c, W_c) energy map.

Same code path is used for real videos and sim composites so train/test
distributions meet in canonical space by construction. That is the whole
point of this module.

Implementation notes
--------------------
- All heavy ops are numpy (jax later if throughput demands). Called at data-loading
  time on CPU; JAX is used only in the training forward pass.
- The band-pass step is a Difference of Gaussians tuned to the target width.
- Width-normalization uses cv2 area/linear resampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


# Chosen from calibration.json p50 (2026-08-28): flagellum widths in real data are
# 4-8 native pixels across sequences. Resampling everything so the flagellum width
# hits 4 canonical pixels means the smallest sequences don't need resampling and the
# largest ones downsample ~2x. Keeps the ViT receptive field consistent.
CANONICAL_TARGET_WIDTH_PX: float = 4.0

# Canonical canvas the encoder ingests. Chosen so that after resampling to
# target-width the cell + a few flagellum lengths fit comfortably.
CANONICAL_H: int = 256
CANONICAL_W: int = 256


@dataclass
class CanonicalConfig:
    """Per-source canonicalization config (usually per-sequence).

    src_width_px: median flagellum width in the SOURCE (native) pixel space. Used
        to compute the resampling factor. From calibration.json.
    bg_median_window: temporal window size (frames) for running median. Use ~2×
        the flagellum beat period for good BG estimation.
    bandpass_sigma_small: DoG small sigma in CANONICAL pixel units (post-resample).
        Roughly half of target flagellum width.
    bandpass_sigma_large: DoG large sigma in canonical units. Kills the low-freq
        residual (any BG structure that survived median subtraction).
    sigma_scale_estimator: 'mad' (robust) or 'std' (simple).
    """
    src_width_px: float = 4.0
    bg_median_window: int = 15
    bandpass_sigma_small: float = 1.5
    bandpass_sigma_large: float = 6.0
    sigma_scale_estimator: str = "mad"
    # Final clipping to keep model inputs bounded (in σ-units, symmetric).
    # 10σ is generous — real flagella maxed around ~150σ in test data before clipping,
    # but the top 0.1% of pixels don't carry useful information beyond "very bright".
    output_clip_sigma: float = 10.0

    @property
    def resample_scale(self) -> float:
        """Multiply source dims by this to reach canonical pixel-per-width."""
        return CANONICAL_TARGET_WIDTH_PX / max(self.src_width_px, 1e-3)


# ---------------------------- Building blocks ----------------------------

def running_median_bg(clip: np.ndarray, window: int) -> np.ndarray:
    """Temporal running median. clip: (T, H, W) float. Returns same shape.

    Uses a rolling window of `window` frames (odd is nicer but not required).
    Edge frames get a truncated window.
    """
    T = clip.shape[0]
    out = np.empty_like(clip, dtype=np.float32)
    half = window // 2
    for t in range(T):
        t0 = max(0, t - half)
        t1 = min(T, t + half + 1)
        out[t] = np.median(clip[t0:t1], axis=0)
    return out


def band_pass_flagellum(residual: np.ndarray, sigma_small: float, sigma_large: float) -> np.ndarray:
    """Difference of Gaussians per frame. residual: (T, H, W) float. Returns same shape.

    (blur_small - blur_large) enhances features at spatial scale ~sigma_small,
    suppresses both fine noise (< sigma_small) and slow gradients (> sigma_large).
    """
    T = residual.shape[0]
    out = np.empty_like(residual, dtype=np.float32)
    for t in range(T):
        s = gaussian_filter(residual[t], sigma=sigma_small)
        l = gaussian_filter(residual[t], sigma=sigma_large)
        out[t] = s - l
    return out


def resample_to_width(clip: np.ndarray, scale: float) -> np.ndarray:
    """Resample each frame's H, W by `scale`. clip: (T, H, W) → (T, H*, W*)."""
    if abs(scale - 1.0) < 1e-3:
        return clip.astype(np.float32)
    T, H, W = clip.shape
    new_h = max(1, int(round(H * scale)))
    new_w = max(1, int(round(W * scale)))
    out = np.empty((T, new_h, new_w), dtype=np.float32)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    for t in range(T):
        out[t] = cv2.resize(clip[t], (new_w, new_h), interpolation=interp)
    return out


def _center_crop_or_pad(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    H, W = frame.shape
    # Pad first if needed
    pad_h = max(0, target_h - H)
    pad_w = max(0, target_w - W)
    if pad_h or pad_w:
        pt, pb = pad_h // 2, pad_h - pad_h // 2
        pl, pr = pad_w // 2, pad_w - pad_w // 2
        frame = np.pad(frame, ((pt, pb), (pl, pr)), mode="constant", constant_values=0.0)
        H, W = frame.shape
    # Center crop if larger
    if H > target_h or W > target_w:
        y0 = (H - target_h) // 2
        x0 = (W - target_w) // 2
        frame = frame[y0 : y0 + target_h, x0 : x0 + target_w]
    return frame


def center_crop_or_pad(clip: np.ndarray, target_h: int = CANONICAL_H, target_w: int = CANONICAL_W) -> np.ndarray:
    """Bring each frame to (target_h, target_w) by center-crop / center-pad."""
    return np.stack([_center_crop_or_pad(clip[t], target_h, target_w) for t in range(clip.shape[0])], axis=0)


def canonical_valid_mask(src_h: int, src_w: int,
                         target_h: int = CANONICAL_H, target_w: int = CANONICAL_W) -> np.ndarray:
    """(target_h, target_w) bool mask: True where a canonical pixel came from real source data.

    `center_crop_or_pad` fills undersized frames with constant 0. Those pixels are NOT
    background — they are padding, and must be excluded from any statistic (σ estimation)
    or any sampling of "real" content (BG-patch harvesting). Mirrors the crop/pad geometry
    of `_center_crop_or_pad` exactly.
    """
    mask = np.ones((src_h, src_w), dtype=bool)
    return _center_crop_or_pad(mask.astype(np.float32), target_h, target_w) > 0.5


def sigma_scale(clip: np.ndarray, estimator: str = "mad",
                valid: Optional[np.ndarray] = None) -> tuple[np.ndarray, float]:
    """Divide the whole clip by a scalar noise estimate. Returns (scaled_clip, σ).

    `valid`: optional (H, W) bool mask. Only these pixels contribute to the estimate.
    Padding pixels are exact zeros; including them collapses the MAD (at >=50% padding
    it hits exactly 0) and every real pixel then blows past the output clip.
    """
    sample = clip if valid is None else clip[:, valid]
    if sample.size == 0:
        raise ValueError("sigma_scale: no valid pixels to estimate σ from")
    if estimator == "mad":
        m = np.median(sample)
        sigma = 1.4826 * np.median(np.abs(sample - m)) + 1e-6
    elif estimator == "std":
        sigma = float(sample.std()) + 1e-6
    else:
        raise ValueError(estimator)
    return clip / sigma, float(sigma)


def temporal_energy_map(clip: np.ndarray, mode: str = "std") -> np.ndarray:
    """Per-pixel energy across time. clip: (T, H, W) → (H, W).

    mode='std': per-pixel std along T. Highlights any pixel that varies across the clip.
    mode='diff': mean |frame_t - frame_{t-1}| along T. Highlights high-frequency motion.
    """
    if mode == "std":
        return clip.std(axis=0)
    if mode == "diff":
        return np.abs(np.diff(clip, axis=0)).mean(axis=0)
    raise ValueError(mode)


# ---------------------------- End-to-end ----------------------------

def canonicalize_clip(
    clip: np.ndarray,
    cfg: CanonicalConfig,
    target_h: int = CANONICAL_H,
    target_w: int = CANONICAL_W,
    center_hint: Optional[tuple[int, int]] = None,
) -> dict:
    """Full pipeline.

    Args:
        clip: (T, H, W) or (T, H, W, C) input. If C>1 the mean over C is used.
        cfg: per-source config (see CanonicalConfig).
        target_h, target_w: output spatial size.
        center_hint: optional (y, x) crop center in POST-RESAMPLE coords. If None,
            center-crop is used. Not yet supported; kept for future cell-detection routing.

    Returns:
        dict with keys:
          - "clip": (T, target_h, target_w) float32 canonical residual (σ-scaled)
          - "valid": (target_h, target_w) bool, True where the pixel is real source
            data rather than center-pad fill
          - "energy": (target_h, target_w) float32 temporal-energy map (in same σ units)
          - "sigma": float, noise σ estimate (canonical scale)
          - "meta": dict with intermediate scales for provenance / debug
    """
    clip = np.asarray(clip, dtype=np.float32)
    if clip.ndim == 4:
        clip = clip.mean(axis=-1)
    assert clip.ndim == 3, f"clip must be (T,H,W), got {clip.shape}"

    # 1. Running-median BG subtract
    bg = running_median_bg(clip, window=cfg.bg_median_window)
    residual = clip - bg

    # 2. Resample to canonical width
    resampled = resample_to_width(residual, cfg.resample_scale)

    # 3. Band-pass (in canonical spatial scale)
    bp = band_pass_flagellum(resampled, cfg.bandpass_sigma_small, cfg.bandpass_sigma_large)

    # 4. Center-crop / pad. `valid` marks which canonical pixels are real source data
    # (vs. constant-0 padding) — everything downstream must respect it.
    cropped = center_crop_or_pad(bp, target_h, target_w)
    valid = canonical_valid_mask(bp.shape[1], bp.shape[2], target_h, target_w)

    # 5. σ-scale (single scalar over the whole clip is fine for now — one video, one σ),
    # estimated over valid pixels ONLY.
    scaled, sigma = sigma_scale(cropped, cfg.sigma_scale_estimator, valid=valid)

    # 5b. Hard clip to output_clip_sigma to keep model inputs bounded. Preserves gradient
    # info near the noise-floor while discarding pathological outliers.
    if cfg.output_clip_sigma > 0:
        scaled = np.clip(scaled, -cfg.output_clip_sigma, cfg.output_clip_sigma)

    # 6. Temporal energy map (in σ units, computed AFTER clipping so it stays bounded)
    energy = temporal_energy_map(scaled, mode="std")

    # Padding carries no signal; keep it at exactly 0 and let `valid` flag it.
    scaled = np.where(valid[None], scaled, 0.0)

    # For the model's static-context input we also expose the RAW temporal
    # median (post-resample, center-cropped/padded like `clip`, but NOT
    # band-passed and NOT σ-scaled). That's the cell body / pipette silhouette
    # the model was trained to attend to.
    static_median_canonical = center_crop_or_pad(
        resample_to_width(np.median(clip, axis=0)[None], cfg.resample_scale)[0][None],
        target_h, target_w)[0]

    return dict(
        clip=scaled.astype(np.float32),
        valid=valid,
        energy=energy.astype(np.float32),
        sigma=sigma,
        static_median=static_median_canonical.astype(np.float32),
        meta=dict(
            src_shape=clip.shape,
            resample_scale=cfg.resample_scale,
            post_resample_shape=resampled.shape,
            post_crop_shape=cropped.shape,
            bg_median_window=cfg.bg_median_window,
        ),
    )
