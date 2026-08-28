"""Tests for the FG-only flagellum simulator."""
import numpy as np

from sim2real.data import (
    CANONICAL_H, CANONICAL_W, FLAGELLUM_K,
    FlagellumSimConfig, sample_scene, SimSampleV2, FlagellumLatent,
)


def _fake_bg(T=16, h=96, w=96, seed=0):
    return np.random.default_rng(seed).standard_normal((T, h, w)).astype(np.float32)


def test_sample_scene_shape():
    rng = np.random.default_rng(0)
    cfg = FlagellumSimConfig(T=16)
    s = sample_scene(rng, cfg, bg_patch=_fake_bg())
    assert isinstance(s, SimSampleV2)
    assert s.clip.shape == (cfg.T, CANONICAL_H, CANONICAL_W)
    assert s.energy.shape == (CANONICAL_H, CANONICAL_W)
    assert s.clip.dtype == np.float32


def test_scene_flagellum_latents_valid():
    rng = np.random.default_rng(0)
    cfg = FlagellumSimConfig(T=8, n_flagella_probs=(0.0, 1.0, 0.0))  # always 1 flagellum
    s = sample_scene(rng, cfg, bg_patch=_fake_bg())
    assert len(s.latents.flagella) == 1
    l = s.latents.flagella[0]
    assert l.attachment.shape == (2,)
    assert l.control_points.shape == (FLAGELLUM_K, 2)
    all_pts = l.all_points()
    assert all_pts.shape == (FLAGELLUM_K + 1, 2)
    assert 0 <= all_pts[0, 0] < CANONICAL_H
    assert 0 <= all_pts[0, 1] < CANONICAL_W


def test_scene_arc_length_positive():
    rng = np.random.default_rng(0)
    cfg = FlagellumSimConfig(T=8, n_flagella_probs=(0.0, 1.0, 0.0))
    s = sample_scene(rng, cfg, bg_patch=_fake_bg())
    l = s.latents.flagella[0]
    L = l.arc_length_px()
    assert cfg.length_min_px * 0.7 <= L <= cfg.length_max_px * 1.4  # loose bound


def test_zero_flagella_case():
    rng = np.random.default_rng(1)
    cfg = FlagellumSimConfig(T=8, n_flagella_probs=(1.0, 0.0, 0.0))  # always 0
    s = sample_scene(rng, cfg, bg_patch=_fake_bg())
    assert len(s.latents.flagella) == 0
    # Clip should be pure BG (numerically the same as the tiled BG); no FG contribution.


def test_two_flagella_case():
    rng = np.random.default_rng(2)
    cfg = FlagellumSimConfig(T=8, n_flagella_probs=(0.0, 0.0, 1.0))  # always 2
    s = sample_scene(rng, cfg, bg_patch=_fake_bg())
    assert len(s.latents.flagella) == 2
    a0 = s.latents.flagella[0].attachment
    a1 = s.latents.flagella[1].attachment
    # Attachments should be separated by ≥30 px (see forbidden_zone in sample_one_flagellum)
    d = float(np.linalg.norm(a0 - a1))
    assert d >= 20  # allow some slack since forbidden radius is exact but samples can still be close if 50 retries fail


def test_no_nan_or_inf():
    rng = np.random.default_rng(3)
    cfg = FlagellumSimConfig(T=8)
    for _ in range(10):
        s = sample_scene(rng, cfg, bg_patch=_fake_bg())
        assert np.isfinite(s.clip).all()
        assert np.isfinite(s.energy).all()


def test_bg_tiling_fills_canvas():
    """Small BG patch should still cover the whole canvas via reflect-tile."""
    rng = np.random.default_rng(4)
    cfg = FlagellumSimConfig(T=8, n_flagella_probs=(1.0, 0.0, 0.0))  # zero flagella so clip = bg
    small_bg = np.ones((cfg.T, 32, 32), dtype=np.float32) * 2.5
    s = sample_scene(rng, cfg, bg_patch=small_bg)
    # Every pixel should be non-zero (BG value 2.5 tiled everywhere)
    assert (s.clip != 0).all()
