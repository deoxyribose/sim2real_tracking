"""Foreground-only flagellum + cell simulator (v1 — 2026-08-28 refactor, cell class added).

For one clip we sample:
  - Presence: 10% empty scene (hard negative), else 1-2 cells with 0-2 flagella each.
  - Each cell: center in canvas, radius, dark amplitude. Rendered as a smooth disk.
  - Each flagellum ATTACHED TO A CELL: attachment lies on that cell's membrane
    (angle + radius), so the model can learn "flagellum starts on cell boundary."
  - Beat parameters per flagellum (amplitude, wavelength, frequency, phase).
  - Roll out T frames of the beating flagellum + static cells.
  - Render as SIGNED intensity contributions (no BG). Composition with a real BG
    patch happens in `render_clip()` at the end.

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
from .types import (
    FLAGELLUM_K, CellLatent, FlagellumLatent, SceneLatents, SimSampleV2,
)


@dataclass
class FlagellumSimConfig:
    T: int = 16
    # High-level scene composition:
    p_empty_scene: float = 0.08                       # hard-negative (no cell, no flagellum)
    n_cells_probs: tuple[float, float] = (0.55, 0.45) # given non-empty: (1 cell, 2 cells)
    n_flagella_per_cell_probs: tuple[float, float, float] = (0.10, 0.45, 0.45)  # 0, 1, 2 flagella per cell

    # ---- Cell body params (canonical px / σ) ----
    # Cell size in real data isn't in calibration.json (we only labeled flagella), but
    # visual inspection of the 256-canvas gives radius roughly 10-70 canonical px. Wetransfer
    # sequences have particularly big cells after width-normalization; go wider than we think
    # we need.
    cell_radius_min_px: float = 10.0
    cell_radius_max_px: float = 70.0
    cell_amplitude_sigma_min: float = 3.0    # cell darkness in σ units
    cell_amplitude_sigma_max: float = 12.0
    cell_margin_px: int = 40                 # keep cell wholly inside the canvas

    # ---- Flagellum params ----
    length_min_px: float = 25.0
    length_max_px: float = 100.0
    width_mean_px: float = 4.0
    width_std_px: float = 0.6
    amplitude_sigma_min: float = 3.0
    amplitude_sigma_max: float = 15.0
    polarity_dark_prob: float = 0.75
    rest_curl_std: float = 0.15
    # Beat: lateral amplitude, wavelength (arc-length cycles), temporal frequency (cycles/T)
    beat_lateral_amp_min: float = 3.0
    beat_lateral_amp_max: float = 10.0
    beat_wave_min: float = 0.5
    beat_wave_max: float = 2.5
    beat_freq_cycles_min: float = 0.5
    beat_freq_cycles_max: float = 3.0
    # When 2 flagella on the same cell: distribute their attachment angles semi-opposite.
    two_flag_angle_between_min: float = 2.0
    two_flag_angle_between_max: float = 3.14


# --------------------------- Cell body rendering ---------------------------

def _render_cell_frame(center_yx: np.ndarray, radius: float, amp: float, H: int, W: int,
                       edge_sigma: float = 1.5) -> np.ndarray:
    """Rasterize a cell body: smooth-edged disk. `amp` is signed magnitude in canonical σ.

    Returns (H, W) float32. Interior peak = -amp (darker than BG), edge softened by edge_sigma.
    """
    yy, xx = np.mgrid[:H, :W]
    d = np.sqrt((yy - center_yx[0]) ** 2 + (xx - center_yx[1]) ** 2)
    # Soft mask: 1 inside, 0 outside, smoothed transition of ~edge_sigma
    inside = 0.5 * (1.0 - np.tanh((d - radius) / max(edge_sigma, 0.5)))
    return (-amp * inside).astype(np.float32)


# --------------------------- Flagellum rendering ---------------------------

def _sample_rest_shape(rng: np.random.Generator, cfg: FlagellumSimConfig,
                       attachment: np.ndarray, base_angle: float) -> tuple[np.ndarray, float]:
    """Sample K arc-length-ordered rest-shape control points."""
    K = FLAGELLUM_K
    L = float(rng.uniform(cfg.length_min_px, cfg.length_max_px))
    step = L / K
    dtheta = rng.normal(0.0, cfg.rest_curl_std, size=K)
    thetas = np.cumsum(dtheta) + base_angle
    steps_y = step * np.sin(thetas)
    steps_x = step * np.cos(thetas)
    positions = np.stack([np.cumsum(steps_y), np.cumsum(steps_x)], axis=1) + attachment[None]
    return positions.astype(np.float32), L


def _apply_beat(control_points: np.ndarray, attachment: np.ndarray, base_angle: float, t_norm: float,
                lat_amp: float, wave: float, freq_cycles: float, phase: float) -> np.ndarray:
    K = control_points.shape[0]
    s = np.linspace(1.0 / K, 1.0, K)
    d = lat_amp * np.sin(2 * np.pi * freq_cycles * t_norm + phase + 2 * np.pi * wave * s)
    perp = np.array([np.cos(base_angle), -np.sin(base_angle)])
    return control_points + d[:, None] * perp[None]


def _render_flagellum_frame(control_points_incl_attach: np.ndarray, width_px: float, amp: float,
                            H: int, W: int, n_curve_samples: int = 64) -> np.ndarray:
    ctrl = control_points_incl_attach
    ts = np.linspace(0, 1, n_curve_samples)
    Kp = ctrl.shape[0]
    seg_ts = ts * (Kp - 1)
    idx_lo = np.clip(np.floor(seg_ts).astype(int), 0, Kp - 2)
    frac = seg_ts - idx_lo
    lo = ctrl[idx_lo]
    hi = ctrl[idx_lo + 1]
    curve = lo * (1 - frac[:, None]) + hi * frac[:, None]

    img = np.zeros((H, W), dtype=np.float32)
    sigma = width_px / 2.5
    r = int(np.ceil(3 * sigma))
    y_range = np.arange(-r, r + 1)
    x_range = np.arange(-r, r + 1)
    yy, xx = np.meshgrid(y_range, x_range, indexing="ij")
    kern = np.exp(-(yy ** 2 + xx ** 2) / (2 * sigma ** 2))
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
    return img * amp


# --------------------------- Scene sampling ---------------------------

def _sample_cell(rng: np.random.Generator, cfg: FlagellumSimConfig,
                 existing_cells: list[CellLatent]) -> Optional[CellLatent]:
    """Sample one cell that doesn't overlap existing cells (best effort)."""
    for _ in range(50):
        r = float(rng.uniform(cfg.cell_radius_min_px, cfg.cell_radius_max_px))
        cy = float(rng.uniform(cfg.cell_margin_px + r, CANONICAL_H - cfg.cell_margin_px - r))
        cx = float(rng.uniform(cfg.cell_margin_px + r, CANONICAL_W - cfg.cell_margin_px - r))
        # Reject if too close to an existing cell (centers within r_new + r_old + 8 px)
        ok = True
        for c in existing_cells:
            if (cy - c.center[0]) ** 2 + (cx - c.center[1]) ** 2 < (r + c.radius_px + 8) ** 2:
                ok = False; break
        if ok:
            amp = float(rng.uniform(cfg.cell_amplitude_sigma_min, cfg.cell_amplitude_sigma_max))
            return CellLatent(center=np.array([cy, cx], dtype=np.float32),
                              radius_px=r, amplitude_sigma=amp)
    return None


def _sample_flagellum_on_cell(rng: np.random.Generator, cfg: FlagellumSimConfig,
                              parent_cell: CellLatent, parent_index: int,
                              used_angles: list[float]) -> tuple[FlagellumLatent, dict]:
    """Sample a flagellum attached to a specific cell's membrane.

    Attachment position = cell.center + cell.radius * (sin θ, cos θ) for a fresh angle θ.
    Prefers angles not already occupied by another flagellum on this cell.
    """
    for _ in range(20):
        theta_attach = float(rng.uniform(0, 2 * np.pi))
        # Reject if too close to an existing flagellum's angle on the same cell (< 60°)
        if all(min(abs(theta_attach - u), 2 * np.pi - abs(theta_attach - u)) > np.pi / 3
               for u in used_angles):
            break
    ay = parent_cell.center[0] + parent_cell.radius_px * np.sin(theta_attach)
    ax = parent_cell.center[1] + parent_cell.radius_px * np.cos(theta_attach)
    attachment = np.array([ay, ax], dtype=np.float32)

    # Base angle of the flagellum: pointing OUTWARD from the cell (attachment - center direction),
    # with slight jitter for natural variation.
    outward = theta_attach + rng.normal(0.0, 0.25)
    control_points, length = _sample_rest_shape(rng, cfg, attachment, outward)
    width = float(np.clip(rng.normal(cfg.width_mean_px, cfg.width_std_px), 2.5, 8.0))
    polarity = int(1 if rng.uniform() < cfg.polarity_dark_prob else -1)
    amp = float(rng.uniform(cfg.amplitude_sigma_min, cfg.amplitude_sigma_max)) * polarity

    beat = dict(
        base_angle=outward,
        lat_amp=float(rng.uniform(cfg.beat_lateral_amp_min, cfg.beat_lateral_amp_max)),
        wave=float(rng.uniform(cfg.beat_wave_min, cfg.beat_wave_max)),
        freq=float(rng.uniform(cfg.beat_freq_cycles_min, cfg.beat_freq_cycles_max)),
        phase=float(rng.uniform(0, 2 * np.pi)),
    )
    latent = FlagellumLatent(
        attachment=attachment, control_points=control_points,
        width_px=width, polarity=polarity, amplitude_sigma=abs(amp),
    )
    used_angles.append(theta_attach)
    return latent, dict(beat=beat, amp=amp, theta_attach=theta_attach,
                        parent_index=parent_index)


# --------------------------- Rendering pipeline ---------------------------

def _render_clip(cfg: FlagellumSimConfig,
                 cells: list[CellLatent],
                 flagella_and_beats: list[tuple[FlagellumLatent, dict]],
                 bg_patch: Optional[np.ndarray]) -> np.ndarray:
    H, W = CANONICAL_H, CANONICAL_W
    # Cells are static across the clip — precompute their combined contribution once.
    cells_contrib = np.zeros((H, W), dtype=np.float32)
    for c in cells:
        cells_contrib += _render_cell_frame(c.center, c.radius_px, c.amplitude_sigma, H, W)

    clip = np.zeros((cfg.T, H, W), dtype=np.float32)
    for t in range(cfg.T):
        t_norm = t / max(cfg.T - 1, 1)
        clip[t] += cells_contrib
        for latent, params in flagella_and_beats:
            beat = params["beat"]
            cp_disp = _apply_beat(latent.control_points, latent.attachment, beat["base_angle"],
                                  t_norm, beat["lat_amp"], beat["wave"], beat["freq"], beat["phase"])
            pts_t = np.concatenate([latent.attachment[None], cp_disp], axis=0)
            clip[t] += _render_flagellum_frame(pts_t, latent.width_px, -params["amp"], H, W)
    if bg_patch is not None:
        clip += _tile_bg(bg_patch, H, W, cfg.T)
    return clip


def _tile_bg(bg_patch: np.ndarray, H: int, W: int, T: int) -> np.ndarray:
    bg_T, bh, bw = bg_patch.shape
    if bg_T < T:
        reps = (T + bg_T - 1) // bg_T
        bg_patch = np.tile(bg_patch, (reps, 1, 1))[:T]
    else:
        bg_patch = bg_patch[:T]
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
    """One clip: sample cells, attach flagella on their membranes, roll out, composite."""
    scene = SceneLatents()
    flagella_and_beats: list[tuple[FlagellumLatent, dict]] = []

    if rng.uniform() >= cfg.p_empty_scene:
        n_cells = int(rng.choice([1, 2], p=cfg.n_cells_probs))
        for _ in range(n_cells):
            c = _sample_cell(rng, cfg, scene.cells)
            if c is not None:
                scene.cells.append(c)

        # Per cell, sample its flagella
        for ci, cell in enumerate(scene.cells):
            n_flag = int(rng.choice([0, 1, 2], p=cfg.n_flagella_per_cell_probs))
            used_angles: list[float] = []
            for _ in range(n_flag):
                latent, params = _sample_flagellum_on_cell(rng, cfg, cell, ci, used_angles)
                # Track parentage in the latent for downstream constraints
                latent_with_parent = FlagellumLatent(
                    attachment=latent.attachment, control_points=latent.control_points,
                    width_px=latent.width_px, polarity=latent.polarity,
                    amplitude_sigma=latent.amplitude_sigma,
                )
                scene.flagella.append(latent_with_parent)
                flagella_and_beats.append((latent_with_parent, params))

    clip = _render_clip(cfg, scene.cells, flagella_and_beats, bg_patch=bg_patch)
    energy = clip.std(axis=0)
    return SimSampleV2(clip=clip.astype(np.float32), energy=energy.astype(np.float32),
                       latents=scene, sigma_canonical=1.0)
