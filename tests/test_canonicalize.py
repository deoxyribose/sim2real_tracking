"""Tests for the canonicalization pipeline."""
import numpy as np
import pytest

from sim2real.data.canonicalize import (
    CANONICAL_H,
    CANONICAL_W,
    CANONICAL_TARGET_WIDTH_PX,
    CanonicalConfig,
    band_pass_flagellum,
    canonicalize_clip,
    center_crop_or_pad,
    resample_to_width,
    running_median_bg,
    sigma_scale,
    temporal_energy_map,
)


def _synthetic_clip(T=16, H=200, W=200, seed=0):
    """A synthetic clip with a slow BG + a moving Gaussian blob (proxy for flagellum)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W]
    bg = 128 + 10 * np.sin(0.03 * xx) + 5 * np.cos(0.02 * yy)
    noise = 3 * rng.standard_normal((T, H, W))
    frames = np.tile(bg[None], (T, 1, 1)).astype(np.float32) + noise
    # Add a small blob that moves horizontally to simulate a "flagellum tip"
    for t in range(T):
        cx = 40 + 5 * t
        cy = H // 2
        sigma = 2.0
        blob = 30 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        frames[t] -= blob  # darker than BG
    return frames


def test_running_median_bg_shape():
    clip = np.random.rand(10, 32, 32).astype(np.float32)
    bg = running_median_bg(clip, window=5)
    assert bg.shape == clip.shape
    assert bg.dtype == np.float32


def test_running_median_bg_kills_dc():
    """If clip is constant across T except for one blob, median should recover the constant."""
    T, H, W = 10, 16, 16
    clip = np.ones((T, H, W), dtype=np.float32) * 5.0
    clip[3, 8, 8] = 100  # single outlier
    bg = running_median_bg(clip, window=5)
    # The middle frame's BG should still be ~5 (outlier gets rejected)
    assert abs(bg[3, 8, 8] - 5.0) < 1e-3


def test_band_pass_reduces_low_freq():
    clip = _synthetic_clip()
    residual = clip - running_median_bg(clip, window=5)
    bp = band_pass_flagellum(residual, sigma_small=1.5, sigma_large=6.0)
    # DC content should drop hard
    assert abs(bp.mean()) < abs(residual.mean()) + 0.1
    # Blob (moving object) should still show up: peak std should be at spatial scale of the blob
    assert bp.std() > 0


def test_resample_scale_2x():
    clip = np.random.rand(4, 100, 80).astype(np.float32)
    out = resample_to_width(clip, scale=2.0)
    assert out.shape == (4, 200, 160)


def test_resample_scale_half():
    clip = np.random.rand(4, 100, 80).astype(np.float32)
    out = resample_to_width(clip, scale=0.5)
    assert out.shape == (4, 50, 40)


def test_resample_scale_identity():
    clip = np.random.rand(4, 100, 80).astype(np.float32)
    out = resample_to_width(clip, scale=1.0)
    assert out.shape == clip.shape
    assert np.allclose(out, clip)


def test_center_crop_bigger_input():
    clip = np.arange(4 * 300 * 300).reshape(4, 300, 300).astype(np.float32)
    out = center_crop_or_pad(clip, 100, 100)
    assert out.shape == (4, 100, 100)


def test_center_pad_smaller_input():
    clip = np.ones((4, 30, 40), dtype=np.float32) * 7
    out = center_crop_or_pad(clip, 100, 100)
    assert out.shape == (4, 100, 100)
    # Center region should contain the original
    y0, x0 = (100 - 30) // 2, (100 - 40) // 2
    assert np.allclose(out[0, y0:y0+30, x0:x0+40], 7.0)
    # Padded edges should be zero
    assert out[0, 0, 0] == 0.0


def test_sigma_scale_normalizes():
    clip = 4.0 * np.random.randn(8, 32, 32).astype(np.float32)
    scaled, sigma = sigma_scale(clip)
    assert abs(sigma - 4.0) < 0.5  # rough
    # After σ-scaling, std should be near 1
    assert 0.7 < scaled.std() < 1.5


def test_temporal_energy_shapes():
    clip = np.random.rand(10, 20, 30).astype(np.float32)
    e_std = temporal_energy_map(clip, mode="std")
    e_diff = temporal_energy_map(clip, mode="diff")
    assert e_std.shape == (20, 30)
    assert e_diff.shape == (20, 30)


def test_canonicalize_end_to_end_shapes():
    clip = _synthetic_clip(T=12, H=200, W=180)
    cfg = CanonicalConfig(src_width_px=4.0)  # scale=1
    out = canonicalize_clip(clip, cfg)
    assert out["clip"].shape == (12, CANONICAL_H, CANONICAL_W)
    assert out["energy"].shape == (CANONICAL_H, CANONICAL_W)
    assert isinstance(out["sigma"], float) and out["sigma"] > 0


def test_canonicalize_resamples_correctly():
    """If src_width_px = 8, we should downsample by ~0.5x before crop."""
    clip = _synthetic_clip(T=8, H=400, W=400)
    cfg = CanonicalConfig(src_width_px=8.0)  # scale=0.5
    assert abs(cfg.resample_scale - 0.5) < 1e-6
    out = canonicalize_clip(clip, cfg)
    # After 0.5x resample, 400x400 becomes 200x200; then crop/pad to 256x256
    assert out["clip"].shape == (8, CANONICAL_H, CANONICAL_W)


def test_canonicalize_energy_highlights_moving_blob():
    """The moving blob should produce non-trivial temporal energy in the energy map."""
    clip = _synthetic_clip(T=16, H=200, W=200)
    cfg = CanonicalConfig(src_width_px=4.0)
    out = canonicalize_clip(clip, cfg)
    e = out["energy"]
    # Energy is not uniformly zero
    assert e.max() > 0.5
    # Max (the blob path) should stand out meaningfully above the median (background noise).
    assert e.max() > 1.5 * np.median(e)


def test_canonicalize_train_test_consistency():
    """Same clip processed twice must give identical output (determinism)."""
    clip = _synthetic_clip()
    cfg = CanonicalConfig(src_width_px=4.0)
    a = canonicalize_clip(clip, cfg)
    b = canonicalize_clip(clip, cfg)
    assert np.allclose(a["clip"], b["clip"])
    assert np.allclose(a["energy"], b["energy"])
