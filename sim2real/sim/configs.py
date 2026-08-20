"""Lightweight config dataclasses for each simulator.

Each simulator's `sample(key, cfg) -> SimSample` reads these. We deliberately keep them as
plain dataclasses (not flax.struct) so they are easy to construct from Python config files
and treat as static under jit.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CommonSimConfig:
    res: int = 128                  # spatial resolution
    T: int = 12                     # number of frames
    n_max: int = 16                 # maximum slots for SimSample; sim must produce ≤ n_max objects
    style_dim: int = 8              # dimensionality of z_style
    obs_noise: float = 0.02         # gaussian noise std on the rendered frame
    perlin_steps: int = 6           # background lattice frequency
    bg_contrast: float = 0.2
    bg_baseline: float = 0.5


@dataclass(frozen=True)
class FlagellaConfig:
    common: CommonSimConfig = field(default_factory=lambda: CommonSimConfig(n_max=8))
    n_objects: int = 4              # how many flagella in this video
    n_points: int = 24              # spline samples per flagellum
    length_min: float = 0.7
    length_max: float = 1.4
    amp_min: float = 0.05
    amp_max: float = 0.15
    freq_min: float = 1.0           # beats per video duration
    freq_max: float = 3.0
    base_jitter: float = 0.02       # tiny per-frame base-point wobble
    line_sigma_px: float = 1.5      # softness of polyline render in pixels


@dataclass(frozen=True)
class ManyCellsConfig:
    common: CommonSimConfig = field(default_factory=lambda: CommonSimConfig(
        n_max=96, bg_baseline=0.4, bg_contrast=0.10,
    ))
    n_objects: int = 25             # was 40 - reduces overlap-blob artefacts in mask sum
    # Bumped radius_min 0.04 -> 0.05 and cell colour so bg range [0.30, 0.50] never
    # touches cell range [0.91, 0.99] — every cell has contrast >= 0.41 pixel-wise.
    radius_min: float = 0.05
    radius_max: float = 0.07
    walk_step: float = 0.015
    cell_color_mean: float = 0.95
    cell_color_std: float = 0.02


@dataclass(frozen=True)
class ManyCellsFastConfig:
    """64x64 x 2 frames, 15 cells. Fast-iteration platform for recipe tuning.

    Radius scaled up to 0.08-0.11 (~5-7 px at 64x64) so cells are still visible.
    n_max only 24 to keep the excess-slot competition modest.
    """
    common: CommonSimConfig = field(default_factory=lambda: CommonSimConfig(
        res=64, T=2, n_max=24, bg_baseline=0.4, bg_contrast=0.10,
    ))
    n_objects: int = 15
    radius_min: float = 0.08
    radius_max: float = 0.11
    walk_step: float = 0.015
    cell_color_mean: float = 0.95
    cell_color_std: float = 0.02


@dataclass(frozen=True)
class ManyCellsFastSmallConfig:
    """64x64 x 2 frames, cells shrunk ~40% vs ManyCellsFast — distribution-shift target
    for fast-platform adaptation experiments. Everything else identical to fast."""
    common: CommonSimConfig = field(default_factory=lambda: CommonSimConfig(
        res=64, T=2, n_max=24, bg_baseline=0.4, bg_contrast=0.10,
    ))
    n_objects: int = 15
    radius_min: float = 0.045
    radius_max: float = 0.065
    walk_step: float = 0.015
    cell_color_mean: float = 0.95
    cell_color_std: float = 0.02


@dataclass(frozen=True)
class ManyCellsSmallConfig:
    """many_cells with cells shrunk ~40% — distribution-shift target for adapt experiments.

    Same generator (many_cells sample), just smaller radius. Everything else (n_objects,
    walk_step, cell colour, background) is identical to ManyCellsConfig so the shift is
    isolated to scale.
    """
    common: CommonSimConfig = field(default_factory=lambda: CommonSimConfig(
        n_max=96, bg_baseline=0.4, bg_contrast=0.10,
    ))
    n_objects: int = 25
    radius_min: float = 0.030
    radius_max: float = 0.045
    walk_step: float = 0.015
    cell_color_mean: float = 0.95
    cell_color_std: float = 0.02


@dataclass(frozen=True)
class MultiScaleConfig:
    common: CommonSimConfig = field(default_factory=lambda: CommonSimConfig(n_max=24))
    n_large: int = 2
    n_small: int = 12
    large_radius_min: float = 0.18
    large_radius_max: float = 0.32
    small_radius_min: float = 0.015
    small_radius_max: float = 0.035
    walk_step_large: float = 0.005
    walk_step_small: float = 0.02


@dataclass(frozen=True)
class TwoCellsConfig:
    """Trivial 2-cell version — diagnostic for whether the arch can decompose ANY multi-object scene."""
    common: CommonSimConfig = field(default_factory=lambda: CommonSimConfig(
        n_max=4, bg_baseline=0.2, bg_contrast=0.0, obs_noise=0.002,
    ))
    n_objects: int = 2
    radius_min: float = 0.09
    radius_max: float = 0.12
    walk_step: float = 0.010
    cell_color_mean: float = 0.98
    cell_color_std: float = 0.005


@dataclass(frozen=True)
class EasyCellsConfig:
    """Trivialised many_cells for A/B testing supervised vs unsupervised.

    Fewer, bigger, brighter, well-separated cells on a UNIFORM background (bg_contrast=0,
    minimal obs noise). If our arch can't decompose THIS unsupervised, arch is the problem.
    """
    common: CommonSimConfig = field(default_factory=lambda: CommonSimConfig(
        n_max=8, bg_baseline=0.2, bg_contrast=0.0, obs_noise=0.002,
    ))
    n_objects: int = 4              # very few cells
    radius_min: float = 0.09        # ~5.8 px on 128²
    radius_max: float = 0.12        # ~7.7 px
    walk_step: float = 0.010
    cell_color_mean: float = 0.98
    cell_color_std: float = 0.005


@dataclass(frozen=True)
class WormsConfig:
    common: CommonSimConfig = field(default_factory=lambda: CommonSimConfig(n_max=12))
    n_objects: int = 6
    n_points: int = 32
    n_ctrl: int = 5                 # control points
    length: float = 0.6             # worm length in normalized coords
    width_px: float = 1.8           # line render sigma
    walk_step: float = 0.01
    rot_step: float = 0.05
