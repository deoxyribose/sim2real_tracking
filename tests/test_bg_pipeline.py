"""Regression tests for the background pipeline.

Covers the three bugs that corrupted `bg_patches_v0.npz`:
  1. center-pad fill harvested as if it were background,
  2. σ estimated over that padding, collapsing the MAD and railing real pixels,
  3. `_tile_bg` mirroring a small patch into a kaleidoscope.
"""
import numpy as np
import pytest

from sim2real.data.canonicalize import (
    CANONICAL_H, CANONICAL_W, CanonicalConfig,
    canonical_valid_mask, canonicalize_clip, sigma_scale,
)
from sim2real.data.sim_flagella import _tile_bg


# ---------------------------- validity mask ----------------------------

def test_valid_mask_marks_pad_region():
    m = canonical_valid_mask(100, 80, 256, 256)
    assert m.shape == (256, 256)
    assert m.sum() == 100 * 80
    # Centred: the valid block sits at rows 78..178, cols 88..168
    ys, xs = np.where(m)
    assert (ys.min(), ys.max()) == (78, 177)
    assert (xs.min(), xs.max()) == (88, 167)


def test_valid_mask_all_true_when_source_is_larger():
    assert canonical_valid_mask(300, 400, 256, 256).all()


# ---------------------------- sigma estimation ----------------------------

@pytest.mark.parametrize("pad_frac", [0.2, 0.4, 0.6])
def test_sigma_ignores_padding(pad_frac):
    """σ must be recovered from the real pixels regardless of how much padding surrounds
    them. Estimating over the padded array is what collapsed the MAD in v0."""
    rng = np.random.default_rng(0)
    H = W = 256
    side = int(round(np.sqrt((1 - pad_frac)) * H))
    valid = canonical_valid_mask(side, side, H, W)
    clip = np.zeros((8, H, W), np.float32)
    clip[:, valid] = (3.0 * rng.standard_normal((8, int(valid.sum())))).astype(np.float32)

    _, sigma_masked = sigma_scale(clip, "mad", valid=valid)
    _, sigma_naive = sigma_scale(clip, "mad")

    assert sigma_masked == pytest.approx(3.0, rel=0.05)
    assert sigma_naive < sigma_masked  # the v0 bug: padding always biases σ downward


def test_padding_does_not_rail_real_pixels():
    """End-to-end: a source frame small enough to need >50% padding must still come out
    with almost nothing on the ±clip rail."""
    rng = np.random.default_rng(1)
    raw = (128 + 4 * rng.standard_normal((16, 150, 150))).astype(np.float32)
    cfg = CanonicalConfig(src_width_px=4.0, bg_median_window=5)
    out = canonicalize_clip(raw, cfg)

    assert out["valid"].shape == (CANONICAL_H, CANONICAL_W)
    assert 0 < out["valid"].mean() < 1, "this fixture should be partly padded"
    # Padding is exactly zero and flagged invalid.
    assert np.all(out["clip"][:, ~out["valid"]] == 0.0)
    # Real pixels are not destroyed by clipping.
    real = out["clip"][:, out["valid"]]
    railed = np.mean(np.abs(real) >= cfg.output_clip_sigma - 1e-3)
    assert railed < 0.01, f"{railed:.1%} of real pixels railed at ±{cfg.output_clip_sigma}σ"


# ---------------------------- background tiling ----------------------------

def test_tile_bg_has_no_mirror_symmetry():
    """The v0 `np.pad(mode='symmetric')` made the canvas mirror-symmetric about its centre."""
    rng = np.random.default_rng(2)
    patch = rng.standard_normal((16, 96, 96)).astype(np.float32)
    canvas = _tile_bg(patch, 256, 256, 16, rng=rng)
    assert canvas.shape == (16, 256, 256)
    f = canvas[0]
    assert not np.allclose(f, f[::-1, :]), "canvas is vertically mirror-symmetric"
    assert not np.allclose(f, f[:, ::-1]), "canvas is horizontally mirror-symmetric"


def test_tile_bg_is_a_plain_crop_when_patch_is_large_enough():
    """Harvesting at canvas size must remove repetition entirely."""
    rng = np.random.default_rng(3)
    patch = rng.standard_normal((16, 300, 300)).astype(np.float32)
    canvas = _tile_bg(patch, 256, 256, 16, rng=rng)
    assert canvas.shape == (16, 256, 256)
    # Every output row must appear verbatim in the source (i.e. it is a crop, not a tiling).
    assert np.isin(canvas[0, 0], patch[0]).all()


def test_tile_bg_varies_across_calls():
    rng = np.random.default_rng(4)
    patch = rng.standard_normal((16, 96, 96)).astype(np.float32)
    a = _tile_bg(patch, 256, 256, 16, rng=np.random.default_rng(10))
    b = _tile_bg(patch, 256, 256, 16, rng=np.random.default_rng(11))
    assert not np.allclose(a, b)


# ---------------------------- harvester ----------------------------

def test_harvester_rejects_padded_candidates():
    """Padding has temporal energy exactly 0, so without the mask it wins the low-energy
    ranking outright. With the mask, no returned centre may touch it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("hbp", "scripts/harvest_bg_patches.py")
    hbp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hbp)

    valid = canonical_valid_mask(160, 160, 256, 256)
    rng = np.random.default_rng(5)
    energy = np.where(valid, 1.0 + 0.1 * rng.standard_normal((256, 256)), 0.0)

    unmasked = hbp.find_low_energy_centers(energy, 96, 96, n=6)
    masked = hbp.find_low_energy_centers(energy, 96, 96, n=6, valid=valid)

    assert unmasked, "fixture should yield candidates without the mask"
    assert any(not valid[y - 48:y + 48, x - 48:x + 48].all() for y, x in unmasked), \
        "without the mask the harvester should be picking padding"
    assert masked, "masked harvest returned nothing"
    for y, x in masked:
        assert valid[y - 48:y + 48, x - 48:x + 48].all()
