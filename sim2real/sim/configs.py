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
    common: CommonSimConfig = field(default_factory=lambda: CommonSimConfig(n_max=64))
    n_objects: int = 40
    radius_min: float = 0.025
    radius_max: float = 0.06
    walk_step: float = 0.015        # per-frame random walk in tanh-space
    cell_color_mean: float = 0.75
    cell_color_std: float = 0.1


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
class WormsConfig:
    common: CommonSimConfig = field(default_factory=lambda: CommonSimConfig(n_max=12))
    n_objects: int = 6
    n_points: int = 32
    n_ctrl: int = 5                 # control points
    length: float = 0.6             # worm length in normalized coords
    width_px: float = 1.8           # line render sigma
    walk_step: float = 0.01
    rot_step: float = 0.05
