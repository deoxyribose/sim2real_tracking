"""Foreground-only flagellum simulator (v0 — flagellum class only).

For one clip we sample:
  - number of flagella: 0 (hard negative), 1, or 2
  - per-flagellum params: attachment, orientation, rest-shape control points,
    beat frequency/amplitude/wavelength/phase, width, polarity, amplitude
  - roll out T frames of the beating flagellum
  - render each frame as the SIGNED intensity contribution of the flagellum(s) —
    NOT a full composite. No background. Composition with a real BG patch happens
    in the training data loader.

Returns SimSampleV2 in canonical space (post-canonicalization).

Ranges are seeded from the 59-annotation calibration (see calibration.json). Every
range is empirical, not guessed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from .canonicalize import CANONICAL_H, CANONICAL_W, CANONICAL_TARGET_WIDTH_PX
from .types import FLAGELLUM_K, FlagellumLatent, SceneLatents, SimSampleV2


# All ranges in CANONICAL pixel/σ units.
@dataclass
class FlagellumSimConfig:
    T: int = 16
    # Number of flagella per clip: probability weights over (0, 1, 2)
    n_flagella_probs: tuple[float, float, float] = (0.10, 0.50, 0.40)   # 10% hard negatives
    # Attachment location (canonical px). Kept inside the canvas with a margin
    attach_margin_px: int = 32
    # Rest shape: total length of the flagellum
    length_min_px: float = 25.0
    length_max_px: float = 100.0
    # Width (canonical px): calibration says median 4, p10-p90 ~ [4, 6]
    width_mean_px: float = 4.0
    width_std_px: float = 0.6
    # Amplitude of the flagellum's signed intensity, in canonical σ units.
    # Calibration: median SNR = 0.37, p90 = 0.73. Amplitude at pixel center is much bigger
    # than SNR (SNR is over the whole mask area). Sample amplitude across sequences seen.
    amplitude_sigma_min: float = 3.0
    amplitude_sigma_max: float = 15.0
    # Polarity: probability that the flagellum is darker than BG (polarity = +1). Real data
    # is mostly darker but polarity+ frac varies 0-0.6 per sequence — use 0.75 as a soft prior.
    polarity_dark_prob: float = 0.75
    # Rest-shape curvature: control-point angular jitter from straight (radians per step)
    rest_curl_std: float = 0.15
    # Beat: lateral amplitude, wavelength (arc-length cycles), temporal frequency (cycles/T)
    beat_lateral_amp_min: float = 3.0
    beat_lateral_amp_max: float = 10.0
    beat_wave_min: float = 0.5
    beat_wave_max: float = 2.5
    beat_freq_cycles_min: float = 0.5
    beat_freq_cycles_max: float = 3.0
    # Two-flagellum symmetry: cells often have two flagella emerging from opposite sides.
    two_flag_angle_between_min: float = 2.0     # radians (opposite = π)
    two_flag_angle_between_max: float = 3.14


def _sample_rest_shape(rng: np.random.Generator, cfg: FlagellumSimConfig,
                       attachment: np.ndarray, base_angle: float) -> tuple[np.ndarray, float]:
    """Sample K arc-length-ordered rest-shape control points relative to `attachment`.

    Returns (control_points_absolute (K, 2), total_length).
    Points are equal-spaced along arc length. The path starts at attachment and grows
    outward in direction `base_angle`, with small per-step angular jitter for curl.
    """
    K = FLAGELLUM_K
    L = float(rng.uniform(cfg.length_min_px, cfg.length_max_px))
    step = L / K
    # Angular walk (small perturbations around base_angle)
    dtheta = rng.normal(0.0, cfg.rest_curl_std, size=K)
    thetas = np.cumsum(dtheta) + base_angle
    steps_y = step * np.sin(thetas)
    steps_x = step * np.cos(thetas)
    positions = np.stack([np.cumsum(steps_y), np.cumsum(steps_x)], axis=1) + attachment[None]
    return positions.astype(np.float32), L


def _apply_beat(control_points: np.ndarray, attachment: np.ndarray, base_angle: float, t_norm: float,
                lat_amp: float, wave: float, freq_cycles: float, phase: float) -> np.ndarray:
    """Add a lateral beat displacement to rest-shape control points.

    Displacement at arc position s ∈ [0, 1] and time t ∈ [0, 1]:
        d(s, t) = lat_amp * sin(2π freq_cycles t + phase + 2π wave s)
    Displacement is perpendicular to `base_angle`.
    """
    K = control_points.shape[0]
    s = np.linspace(1.0 / K, 1.0, K)   # arc-length fraction of each control point
    d = lat_amp * np.sin(2 * np.pi * freq_cycles * t_norm + phase + 2 * np.pi * wave * s)
    # Perpendicular direction to base_angle
    perp = np.array([np.cos(base_angle), -np.sin(base_angle)])  # (dy, dx) rotated 90° from (sin, cos)
    return control_points + d[:, None] * perp[None]


def _render_flagellum_frame(control_points_incl_attach: np.ndarray, width_px: float, amp: float,
                            H: int, W: int, n_curve_samples: int = 64) -> np.ndarray:
    """Rasterize a flagellum onto a (H, W) canvas.

    control_points_incl_attach: (K+1, 2) float, (y, x). Point 0 = attachment.
    Returns (H, W) float32 signed contribution. Values are approx in [-amp, amp].
    """
    # Densify the curve with linear interpolation between control points
    ctrl = control_points_incl_attach
    ts = np.linspace(0, 1, n_curve_samples)
    Kp = ctrl.shape[0]
    # Piecewise-linear interp along the control points
    seg_ts = ts * (Kp - 1)
    idx_lo = np.clip(np.floor(seg_ts).astype(int), 0, Kp - 2)
    frac = seg_ts - idx_lo
    lo = ctrl[idx_lo]
    hi = ctrl[idx_lo + 1]
    curve = lo * (1 - frac[:, None]) + hi * frac[:, None]  # (n_curve_samples, 2)

    # Stamp Gaussians along the curve
    img = np.zeros((H, W), dtype=np.float32)
    sigma = width_px / 2.5  # width_px roughly = 2.5σ full width at appreciable amplitude
    r = int(np.ceil(3 * sigma))  # kernel radius
    y_range = np.arange(-r, r + 1)
    x_range = np.arange(-r, r + 1)
    yy, xx = np.meshgrid(y_range, x_range, indexing="ij")
    kern = np.exp(-(yy ** 2 + xx ** 2) / (2 * sigma ** 2))  # peak = 1
    # Stamp — accumulate with MAX rather than sum, so overlap doesn't over-darken.
    for (py, px) in curve:
        iy, ix = int(round(py)), int(round(px))
        y0, y1 = max(0, iy - r), min(H, iy + r + 1)
        x0, x1 = max(0, ix - r), min(W, ix + r + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        ky0 = y0 - (iy - r)
        kx0 = x0 - (ix - r)
        ky1 = ky0 + (y1 - y0)
        kx1 = kx0 + (x1 - x0)
        stamp = kern[ky0:ky1, kx0:kx1]
        img[y0:y1, x0:x1] = np.maximum(img[y0:y1, x0:x1], stamp)
    return img * amp    # signed amplitude


def sample_one_flagellum(rng: np.random.Generator, cfg: FlagellumSimConfig,
                         forbidden_attachment_zones: list[tuple[float, float, float]] = ()
                         ) -> tuple[FlagellumLatent, dict]:
    """Sample one flagellum's static latents + beat params.

    forbidden_attachment_zones: list of (cy, cx, radius) — don't put attachment inside these.
    Returns (FlagellumLatent, beat_params_dict).
    """
    for _ in range(50):
        ay = rng.uniform(cfg.attach_margin_px, CANONICAL_H - cfg.attach_margin_px)
        ax = rng.uniform(cfg.attach_margin_px, CANONICAL_W - cfg.attach_margin_px)
        if not any((ay - cy) ** 2 + (ax - cx) ** 2 < r ** 2 for (cy, cx, r) in forbidden_attachment_zones):
            break
    attachment = np.array([ay, ax], dtype=np.float32)

    base_angle = float(rng.uniform(0, 2 * np.pi))
    control_points, length = _sample_rest_shape(rng, cfg, attachment, base_angle)
    width = float(np.clip(rng.normal(cfg.width_mean_px, cfg.width_std_px), 2.5, 8.0))
    polarity = int(1 if rng.uniform() < cfg.polarity_dark_prob else -1)
    amp = float(rng.uniform(cfg.amplitude_sigma_min, cfg.amplitude_sigma_max)) * polarity

    beat = dict(
        base_angle=base_angle,
        lat_amp=float(rng.uniform(cfg.beat_lateral_amp_min, cfg.beat_lateral_amp_max)),
        wave=float(rng.uniform(cfg.beat_wave_min, cfg.beat_wave_max)),
        freq=float(rng.uniform(cfg.beat_freq_cycles_min, cfg.beat_freq_cycles_max)),
        phase=float(rng.uniform(0, 2 * np.pi)),
    )
    latent = FlagellumLatent(
        attachment=attachment,
        control_points=control_points,
        width_px=width,
        polarity=polarity,
        amplitude_sigma=abs(amp),
    )
    return latent, dict(beat=beat, amp=amp)


def render_clip(cfg: FlagellumSimConfig, latents_and_beats: list[tuple[FlagellumLatent, dict]],
                bg_patch: Optional[np.ndarray] = None) -> np.ndarray:
    """Roll out T frames and render each flagellum's signed intensity.

    If bg_patch is provided (T, H, W) or (T, H_bg, W_bg) with H_bg <= H_c, it's added
    behind the FG (composite = bg + FG). Otherwise the returned clip is FG-only.
    """
    H, W = CANONICAL_H, CANONICAL_W
    clip = np.zeros((cfg.T, H, W), dtype=np.float32)
    for t in range(cfg.T):
        t_norm = t / max(cfg.T - 1, 1)
        for latent, params in latents_and_beats:
            beat = params["beat"]
            # Represent the "clip midpoint" pose as the LATENT (this is what the model predicts).
            # For time t we need to displace by the CHANGE from midpoint.
            def displace(t_target):
                pts_incl = np.concatenate([latent.attachment[None], latent.control_points], axis=0)
                cp_disp = _apply_beat(latent.control_points, latent.attachment, beat["base_angle"],
                                      t_target, beat["lat_amp"], beat["wave"], beat["freq"], beat["phase"])
                return np.concatenate([latent.attachment[None], cp_disp], axis=0)

            pts_t = displace(t_norm)
            frame_fg = _render_flagellum_frame(pts_t, latent.width_px, -params["amp"], H, W)
            # amp sign: params["amp"] is signed (polarity applied); we want darker = negative pixel value
            # in canonical space (since canonicalization made darker-than-BG negative-valued residuals).
            clip[t] += frame_fg
    if bg_patch is not None:
        clip += _tile_bg(bg_patch, H, W, cfg.T)
    return clip


def _tile_bg(bg_patch: np.ndarray, H: int, W: int, T: int) -> np.ndarray:
    """Fill a (T, H, W) canvas with a (T, h, w) BG patch — reflect-tiled if smaller.

    Ensures BG covers every pixel so composited flagella never sit on zero-background.
    Tiles in time by wrap. Reflect-pads in space using np.pad(mode='symmetric').
    """
    bg_T, bh, bw = bg_patch.shape
    # Time: wrap
    if bg_T < T:
        reps = (T + bg_T - 1) // bg_T
        bg_patch = np.tile(bg_patch, (reps, 1, 1))[:T]
    else:
        bg_patch = bg_patch[:T]
    # Space: if smaller than canvas, reflect-pad; if larger, center-crop
    if bh < H or bw < W:
        pad_h = max(0, H - bh)
        pad_w = max(0, W - bw)
        pt, pb = pad_h // 2, pad_h - pad_h // 2
        pl, pr = pad_w // 2, pad_w - pad_w // 2
        bg_patch = np.pad(bg_patch, ((0, 0), (pt, pb), (pl, pr)), mode="symmetric")
    if bg_patch.shape[1] > H or bg_patch.shape[2] > W:
        by0 = (bg_patch.shape[1] - H) // 2
        bx0 = (bg_patch.shape[2] - W) // 2
        bg_patch = bg_patch[:, by0 : by0 + H, bx0 : bx0 + W]
    return bg_patch.astype(np.float32)


def sample_scene(rng: np.random.Generator, cfg: FlagellumSimConfig,
                 bg_patch: Optional[np.ndarray] = None) -> SimSampleV2:
    """One clip: sample scene, roll out, return SimSampleV2 with GT latents."""
    n_flag = int(rng.choice([0, 1, 2], p=cfg.n_flagella_probs))

    latents_and_beats: list[tuple[FlagellumLatent, dict]] = []
    forbidden: list[tuple[float, float, float]] = []
    for _ in range(n_flag):
        latent, beat = sample_one_flagellum(rng, cfg, forbidden)
        latents_and_beats.append((latent, beat))
        # Keep flagella away from each other's attachments (cells only 1-2 flagella, both from
        # opposite sides of the same cell body). Add a soft exclusion zone.
        forbidden.append((float(latent.attachment[0]), float(latent.attachment[1]), 30.0))

    clip = render_clip(cfg, latents_and_beats, bg_patch=bg_patch)
    # Compute energy map on the rendered clip
    energy = clip.std(axis=0)

    scene = SceneLatents(flagella=[l for l, _ in latents_and_beats])
    # σ estimate: after canonicalization real clips are ~unit-σ. Simulated BG is real,
    # so σ ~ 1 by construction; if no BG was provided, σ is degenerate — record as 1.0.
    return SimSampleV2(clip=clip.astype(np.float32), energy=energy.astype(np.float32),
                       latents=scene, sigma_canonical=1.0)
