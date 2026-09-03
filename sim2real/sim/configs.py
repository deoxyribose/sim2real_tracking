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
class FlagellaFastConfig:
    """64x64 x 2 frames, 3 flagella — fast platform for the winning worm recipe test."""
    common: CommonSimConfig = field(default_factory=lambda: CommonSimConfig(
        res=64, T=2, n_max=6,
    ))
    n_objects: int = 3
    n_points: int = 24
    length_min: float = 0.7
    length_max: float = 1.4
    amp_min: float = 0.05
    amp_max: float = 0.15
    freq_min: float = 1.0
    freq_max: float = 3.0
    base_jitter: float = 0.02
    line_sigma_px: float = 0.75      # halved for 64x64


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
class AlgaeConfig:
    """Chlamydomonas-like: one round cell body + two beating flagella.

    3 slot objects per video: body (slot 0), flagellum 1 (slot 1), flagellum 2 (slot 2).
    Bright background, dark features (opposite polarity from most other sims here — matches
    the real algae_data/ frames).
    """
    common: CommonSimConfig = field(default_factory=lambda: CommonSimConfig(
        res=128, T=8, n_max=6, bg_baseline=0.55, bg_contrast=0.05, obs_noise=0.015,
    ))
    # Body sizes span a bimodal-ish real distribution: small free-floating cells (~2-3% area)
    # and larger pipette-held cells (~10-20% area). Widened range covers both without needing
    # explicit modes. Auto-seg on annotated frames biased toward small; visual inspection of
    # pipette-mounted cells shows body diameter ~30-40% of frame width → rx ~ 0.30-0.40.
    body_rx_min: float = 0.16
    body_rx_max: float = 0.36
    body_ry_min: float = 0.14
    body_ry_max: float = 0.34
    body_edge_softness: float = 0.008
    body_jitter: float = 0.006
    body_halo_width_min: float = 0.06
    body_halo_width_max: float = 0.14
    body_halo_intensity_min: float = 0.08
    body_halo_intensity_max: float = 0.20
    # Real body interior lands near 0.20 while bg ~0.75; that requires darkening ≈ 0.55.
    # Widened up from prior 0.10-0.20 which produced grey bodies.
    body_dark_intensity_min: float = 0.30
    body_dark_intensity_max: float = 0.60
    body_edge_darkening_min: float = 0.10
    body_edge_darkening_max: float = 0.22
    anchor_angle_offset: float = 0.5             # rad, ~28°
    # Flagellum length: real p90 = 0.32 of image width. Sim length is in [-1, 1] half-extent,
    # so 0.32 image-frac × 2 = 0.64 sim units. Extended max from 0.45 → 0.70 to capture the tail.
    flag_length_min: float = 0.15
    flag_length_max: float = 0.70
    # Wider amplitude + higher wave number to reproduce the very curly recovery-stroke
    # flagella seen in real frames (real straightness p10 = 0.73; old sim never below 0.96).
    flag_amp_min: float = 0.04
    flag_amp_max: float = 0.16
    flag_freq_min: float = 1.0
    flag_freq_max: float = 3.0
    flag_k_min: float = 4.0
    flag_k_max: float = 16.0
    # Bigger swing amplitude — some annotated flagella extend nearly perpendicular to body axis.
    swing_amp_min: float = 0.20
    swing_amp_max: float = 0.90
    n_points: int = 32
    # Sim's measured flagella thickness (3.5 px @ 128) was ~50% too thick vs real (2.3 px);
    # lowered max sigma to bring the distribution down.
    line_sigma_px_min: float = 0.6
    line_sigma_px_max: float = 1.2
    # Flagella are BRIGHT in real phase-contrast data (bright thin lines against grey bg).
    flag_bright_intensity_min: float = 0.10
    flag_bright_intensity_max: float = 0.25
    # Fraction of videos with only one visible flagellum (real: 22% of frames show a single
    # flagellum — occlusion by pipette or recovery-stroke overlap with body). When triggered,
    # slot 2's z_pres AND mask go to zero for the whole video.
    single_flag_prob: float = 0.22
    # Fraction of videos where flagella render DARKER than bg (rest are lighter). Real data
    # shows both polarities depending on focus/contrast; user-reported ~30% dark cases.
    dark_flag_prob: float = 0.30
    # Per-video motion-blur σ (pixels) applied to the flagella mask before render. Fast beating
    # → wider + fainter trace in the recorded frame. 0 = crisp, 3 = heavily blurred.
    flag_motion_blur_min: float = 0.0
    flag_motion_blur_max: float = 2.5
    # DoG (difference-of-gaussians) internal-contour amplitude for body & pipette. Adds
    # phase-contrast-like ring structure inside dense objects — matches real chloroplast /
    # pipette wall reflections.
    body_dog_amp_min: float = 0.10
    body_dog_amp_max: float = 0.30
    pipette_dog_amp_min: float = 0.15
    pipette_dog_amp_max: float = 0.35
    # Pipette (slot 3): phase-contrast bright interior + dark edge rims. Static per video.
    pipette_prob: float = 0.55
    pipette_bar_half_min: float = 0.08
    pipette_bar_half_max: float = 0.18
    # Dark rims at the glass-wall edges (subtracted from frame at |dy|=local_half_width).
    pipette_edge_dark_min: float = 0.15
    pipette_edge_dark_max: float = 0.40
    # Bright interior (added to frame inside the bar) — matches phase-contrast optics.
    pipette_interior_bright_min: float = 0.06
    pipette_interior_bright_max: float = 0.18
    # Per-video global imaging conditions (applied to the final rendered frame).
    img_brightness_min: float = -0.15
    img_brightness_max: float = 0.15
    img_contrast_min: float = 0.70
    img_contrast_max: float = 1.30
    img_gamma_min: float = 0.70
    img_gamma_max: float = 1.40
    img_blur_sigma_min: float = 0.0
    img_blur_sigma_max: float = 1.2
    # Real bg mean ≈ 0.77 in annotated data (bright phase-contrast). Old sim was 0.50-0.58.
    bg_baseline_min: float = 0.65
    bg_baseline_max: float = 0.85
    bg_variation: float = 0.03
    n_bg_particles_min: int = 40
    n_bg_particles_max: int = 100
    bg_particle_size: float = 0.008
    bg_particle_size_variance: float = 5.0
    bg_particle_intensity_min: float = 0.10
    bg_particle_intensity_max: float = 0.20
    bg_hf_noise_min: float = 0.010
    bg_hf_noise_max: float = 0.030
    # Body interior texture (chloroplast-like structure) — per-video sampled amplitude
    body_texture_amp_min: float = 0.03
    body_texture_amp_max: float = 0.10
    body_texture_freq: int = 10                  # Perlin steps — higher = finer texture
    feature_intensity: float = 0.10               # legacy


@dataclass(frozen=True)
class AlgaeFastConfig:
    """64x64 x 2 frames algae — fast platform for the algae experiments."""
    common: CommonSimConfig = field(default_factory=lambda: CommonSimConfig(
        res=64, T=2, n_max=6, bg_baseline=0.55, bg_contrast=0.05, obs_noise=0.015,
    ))
    body_rx_min: float = 0.16
    body_rx_max: float = 0.36
    body_ry_min: float = 0.14
    body_ry_max: float = 0.34
    body_edge_softness: float = 0.008
    body_jitter: float = 0.006
    body_halo_width_min: float = 0.06
    body_halo_width_max: float = 0.14
    body_halo_intensity_min: float = 0.08
    body_halo_intensity_max: float = 0.20
    body_dark_intensity_min: float = 0.30
    body_dark_intensity_max: float = 0.60
    body_edge_darkening_min: float = 0.10
    body_edge_darkening_max: float = 0.22
    anchor_angle_offset: float = 0.5
    flag_length_min: float = 0.15
    flag_length_max: float = 0.70
    flag_amp_min: float = 0.04
    flag_amp_max: float = 0.16
    flag_freq_min: float = 1.0
    flag_freq_max: float = 3.0
    flag_k_min: float = 4.0
    flag_k_max: float = 16.0
    swing_amp_min: float = 0.20
    swing_amp_max: float = 0.90
    n_points: int = 32
    line_sigma_px_min: float = 0.3
    line_sigma_px_max: float = 0.6
    flag_bright_intensity_min: float = 0.10
    flag_bright_intensity_max: float = 0.25
    single_flag_prob: float = 0.22
    dark_flag_prob: float = 0.30
    flag_motion_blur_min: float = 0.0
    flag_motion_blur_max: float = 2.5
    body_dog_amp_min: float = 0.10
    body_dog_amp_max: float = 0.30
    pipette_dog_amp_min: float = 0.15
    pipette_dog_amp_max: float = 0.35
    pipette_prob: float = 0.55
    pipette_bar_half_min: float = 0.08
    pipette_bar_half_max: float = 0.18
    pipette_edge_dark_min: float = 0.15
    pipette_edge_dark_max: float = 0.40
    pipette_interior_bright_min: float = 0.06
    pipette_interior_bright_max: float = 0.18
    img_brightness_min: float = -0.15
    img_brightness_max: float = 0.15
    img_contrast_min: float = 0.70
    img_contrast_max: float = 1.30
    img_gamma_min: float = 0.70
    img_gamma_max: float = 1.40
    img_blur_sigma_min: float = 0.0
    img_blur_sigma_max: float = 1.2
    bg_baseline_min: float = 0.65
    bg_baseline_max: float = 0.85
    bg_variation: float = 0.03
    n_bg_particles_min: int = 15
    n_bg_particles_max: int = 40
    bg_particle_size: float = 0.012
    bg_particle_size_variance: float = 5.0
    bg_particle_intensity_min: float = 0.10
    bg_particle_intensity_max: float = 0.20
    bg_hf_noise_min: float = 0.010
    bg_hf_noise_max: float = 0.030
    body_texture_amp_min: float = 0.03
    body_texture_amp_max: float = 0.10
    body_texture_freq: int = 8
    feature_intensity: float = 0.10


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


@dataclass(frozen=True)
class WormsFastConfig:
    """64x64 x 2 frames, 4 worms — fast-iteration platform for shape-sim experiments.

    width_px scaled down (1.8 -> 0.9) since resolution halved. length in normalized coords
    unchanged (0.6 = fraction of image). Fewer worms (4 vs 6) to reduce overlap on small canvas.
    """
    common: CommonSimConfig = field(default_factory=lambda: CommonSimConfig(
        res=64, T=2, n_max=8,
    ))
    n_objects: int = 4
    n_points: int = 32
    n_ctrl: int = 5
    length: float = 0.6
    width_px: float = 0.9
    walk_step: float = 0.01
    rot_step: float = 0.05
