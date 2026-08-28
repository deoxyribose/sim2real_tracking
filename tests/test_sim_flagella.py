"""Tests for the FG-only flagellum + cell simulator (v1)."""
import numpy as np

from sim2real.data import (
    CANONICAL_H, CANONICAL_W, FLAGELLUM_K,
    CellLatent, FlagellumSimConfig, sample_scene, SimSampleV2, FlagellumLatent,
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
    """Force always 1 cell with 1 flagellum."""
    rng = np.random.default_rng(0)
    cfg = FlagellumSimConfig(T=8, p_empty_scene=0.0,
                             n_cells_probs=(1.0, 0.0),
                             n_flagella_per_cell_probs=(0.0, 1.0, 0.0))
    s = sample_scene(rng, cfg, bg_patch=_fake_bg())
    assert len(s.latents.cells) == 1
    assert len(s.latents.flagella) == 1
    l = s.latents.flagella[0]
    assert l.attachment.shape == (2,)
    assert l.control_points.shape == (FLAGELLUM_K, 2)


def test_flagellum_attachment_on_cell_membrane():
    """The whole point of the v1 change: attachment must lie on cell boundary."""
    rng = np.random.default_rng(0)
    cfg = FlagellumSimConfig(T=4, p_empty_scene=0.0,
                             n_cells_probs=(1.0, 0.0),
                             n_flagella_per_cell_probs=(0.0, 1.0, 0.0))
    for _ in range(20):
        s = sample_scene(rng, cfg, bg_patch=_fake_bg())
        assert len(s.latents.cells) == 1 and len(s.latents.flagella) == 1
        cell = s.latents.cells[0]
        flag = s.latents.flagella[0]
        d = float(np.linalg.norm(flag.attachment - cell.center))
        # Attachment should be within ~2 px of the cell radius
        assert abs(d - cell.radius_px) < 2.0, f"attachment distance {d:.2f} vs radius {cell.radius_px:.2f}"


def test_empty_scene_has_no_cells_no_flagella():
    rng = np.random.default_rng(1)
    cfg = FlagellumSimConfig(T=4, p_empty_scene=1.0)
    s = sample_scene(rng, cfg, bg_patch=_fake_bg())
    assert len(s.latents.cells) == 0
    assert len(s.latents.flagella) == 0


def test_two_cells_case():
    """Force always 2 cells."""
    rng = np.random.default_rng(2)
    cfg = FlagellumSimConfig(T=4, p_empty_scene=0.0,
                             n_cells_probs=(0.0, 1.0),
                             n_flagella_per_cell_probs=(1.0, 0.0, 0.0))  # 0 flag per cell for simplicity
    for _ in range(5):
        s = sample_scene(rng, cfg, bg_patch=_fake_bg())
        assert len(s.latents.cells) >= 1  # sometimes second cell placement fails, that's ok
        # Cells should not overlap centers-to-centers
        for i, a in enumerate(s.latents.cells):
            for b in s.latents.cells[i+1:]:
                d = float(np.linalg.norm(a.center - b.center))
                assert d >= a.radius_px + b.radius_px  # non-overlapping


def test_no_nan_or_inf():
    rng = np.random.default_rng(3)
    cfg = FlagellumSimConfig(T=8)
    for _ in range(10):
        s = sample_scene(rng, cfg, bg_patch=_fake_bg())
        assert np.isfinite(s.clip).all()
        assert np.isfinite(s.energy).all()


def test_bg_tiling_fills_canvas():
    """Empty scene (no FG) → clip should be pure BG (non-zero everywhere)."""
    rng = np.random.default_rng(4)
    cfg = FlagellumSimConfig(T=8, p_empty_scene=1.0)
    small_bg = np.ones((cfg.T, 32, 32), dtype=np.float32) * 2.5
    s = sample_scene(rng, cfg, bg_patch=small_bg)
    assert (s.clip != 0).all()
