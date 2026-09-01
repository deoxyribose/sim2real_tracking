"""Diverse flagellum simulator (JAX).

Goals
-----
1. Higher-diversity dynamics: multiple beat modes drawn from biophysics —
   curvature wave (physical), tip-amplified (sperm-like Machin damping),
   breaststroke (Chlamydomonas asymmetric 2-harmonic), near-static bent,
   and the legacy lateral-sine baseline for coverage.
2. Higher-diversity geometry: cells with soft-repulsion contact so they
   don't overlap, flagella attached to cell membranes (both polarities),
   plus free-floating short debris-flagella.
3. Higher-diversity appearance: per-clip faintness, defocus blur, motion
   blur, drifting debris blobs, vignette, additive noise.
4. Both polarities: flagella can be darker OR lighter than background.
5. Output: raw composite clip and temporal-median-subtracted clip
   (residual is what the model actually sees).

All arrays are fixed-shape (n_max_cells + n_max_flagella slots, alive
via a presence mask) so the whole `sample_clip` call is jit + vmap-able.

Physics shortcut
----------------
Rather than integrating an elastic-rod ODE we prescribe a curvature wave
`κ(s,t)` and integrate it to `(x, y)` — one cumulative sum, no solver.
Cell-cell contact is Jacobi soft-repulsion (a few steps of gradient
descent on pairwise overlap). No flagellum-cell interaction: cheap and
the visual is fine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp

from sim2real.sim.background import perlin_noise
from sim2real.sim.splines import render_polyline_segments_sdf

Array = jnp.ndarray


# ---- Beat modes -----------------------------------------------------------

MODE_CURV_WAVE = 0     # planar curvature wave κ(s,t)=A·sin(2π(ks−ft)+φ)
MODE_TIP_AMP = 1       # tip-amplified: A(s) = A · (s/L)^α
MODE_BREASTSTROKE = 2  # asymmetric 2-harmonic curvature wave
MODE_STATIC_BENT = 3   # fixed curved rest shape + tiny flicker
MODE_LATERAL_SINE = 4  # legacy: straight rod + lateral sine displacement
N_MODES = 5


# ---- Config ---------------------------------------------------------------

@dataclass(frozen=True)
class DiverseSimConfig:
    # Frame layout
    T: int = 32
    H: int = 256
    W: int = 256

    # Slot capacity (fixed shapes for jit). Real data almost never shows more
    # than 2 flagella per frame and rarely 2 cells simultaneously.
    n_max_cells: int = 2
    n_max_flagella: int = 4
    n_arc_pts: int = 24                     # arc-length samples per flagellum

    # Scene composition (mixture over active counts)
    #   probs indexed 0..len-1; e.g. (0.85, 0.15) → P(1)=.85, P(2)=.15
    # Real: mostly 1 cell in view; 2 cells rare.
    n_cells_probs: tuple = (0.85, 0.15)
    # Real per-cell distribution: mostly 2 flagella (Chlamydomonas biology),
    # sometimes 1. Never 0 (empty scenes waste model capacity) and never >2.
    n_flag_per_cell_probs: tuple = (0.0, 0.20, 0.80)
    # Real data has no free-floating flagella; keep at 0.
    n_free_flag_probs: tuple = (1.0,)

    # Cell body ranges (canonical pixels). Real body labels @ 512-canvas:
    # major axis p50=56, minor p50=50 → on 200-canvas the cell body is ~10 px
    # radius on average. Wider range for coverage across zoom levels.
    cell_radius_min: float = 8.0
    cell_radius_max: float = 30.0
    # Axis ratio (minor/major). Real p10-p90 = 0.89-0.92; extend to 0.65-1.0
    # so the sim covers elongated / rounder cells too.
    cell_axis_ratio_min: float = 0.65
    cell_axis_ratio_max: float = 1.00
    # Real cell/BG intensity gap is ~0.10-0.20 in [0,1] units.
    cell_amp_min: float = 0.05
    cell_amp_max: float = 0.28
    cell_margin: float = 30.0
    cell_edge_sigma: float = 2.0

    # Flagellum geometry. Real labels @ 512-canvas: width p50=10, length p50=86.
    # Scaled to our 200-canvas: width ~4 px, length ~34 px.
    flag_length_min: float = 20.0
    flag_length_max: float = 80.0
    # SDF sigma; visible width ≈ 2×σ. Target ~2-4 px visible → σ ∈ [0.8, 2.0].
    flag_width_min: float = 0.8
    flag_width_max: float = 2.0
    # Real residual p99.9 spans 0.02-0.44 across sequences; extend a bit either
    # side. Bumped max to 0.70 to cover the p99.99 tail (0.75 on rare bright pixels).
    # `flag_amp_bias > 1` biases sampling toward the LOW end: amp is drawn as
    # amp_min + (amp_max − amp_min) · u^bias  for u ~ Uniform(0,1).
    # bias=2 gives median amp near 0.19 instead of 0.36 (uniform) — matches
    # real videos where most flagella are quite faint.
    flag_amp_min: float = 0.02
    flag_amp_max: float = 0.70
    flag_amp_bias: float = 2.5
    flag_dark_prob: float = 0.55            # P(polarity == -1, i.e. darker than BG)

    # Beat mode mixture. Real labels show peak curvature at ~0.88 along the
    # arc (near tip), median angle sweep ~55°. That's the tip-amplified
    # signature → bump TIP_AMP from 10% → 40%. κwave (uniform) is not what
    # real does; drop to 15%. Static_bent keeps large weight for smooth arcs.
    mode_weights: tuple = (0.15, 0.40, 0.10, 0.30, 0.05)

    # Curvature-wave params. `curv_amp` is peak radians per s-unit (s ∈ [0,1]).
    # Real labels show chord-deviation ≈ 13% of arc length → total angle sweep
    # ~π/6. With k≈0.5 that means amp ~1.5 typical; extend range for coverage.
    curv_amp_min: float = 0.5
    curv_amp_max: float = 3.0
    curv_k_min: float = 0.2                 # spatial cycles along filament
    curv_k_max: float = 1.0                 # ~one bend along the length max
    # Temporal cycles per clip. Widened to 0.2-3.0 for more visible beat-rate
    # variation across clips (some flagella whipping, others slow). Motion
    # blur is capped at 2 frames so aliasing from higher freqs is limited.
    curv_freq_min: float = 0.2
    curv_freq_max: float = 3.0

    # Tip-amplification exponent
    tip_alpha_min: float = 1.0
    tip_alpha_max: float = 2.5

    # Breaststroke 2nd harmonic weight (reduced — real flagella rarely show
    # sharp asymmetric bends within a single frame).
    breast_h2_min: float = 0.10
    breast_h2_max: float = 0.35

    # Static bent: peak κ(s). Kept small — real static flagella are gently curved.
    static_bend_min: float = 0.2
    static_bend_max: float = 1.5
    static_flicker: float = 0.05

    # Lateral (baseline) — reduced from legacy sim (was 2-12 → too wavy).
    lat_amp_min: float = 0.5
    lat_amp_max: float = 3.0
    lat_wave_min: float = 0.3
    lat_wave_max: float = 1.2
    lat_freq_min: float = 0.5
    lat_freq_max: float = 2.5

    # Cell contact-force relaxation (real cells never overlap).
    contact_iters: int = 24
    contact_margin: float = 12.0            # extra px space between cell edges
    contact_lr: float = 0.6

    # Slow cell drift (linear translation per clip). Real videos show clear
    # cell/pipette drift over a few-second window — creates the red/blue halo
    # in the residual that the model needs to be robust to.
    cell_drift_speed_max: float = 3.0       # total px drift over the whole clip

    # Per-clip scene scale: multiplies cell radius + flagellum length + flag
    # width + pipette dims (but NOT distortions/noise/drift). Real videos span
    # zoomed-in (big cell fills frame) to zoomed-out. Was 0.4-1.6 → the low
    # end made cells 3-4 px radius, indistinguishable from noise/flagella.
    scene_scale_min: float = 0.65
    scene_scale_max: float = 1.7

    # Pipette: HOLLOW + CONICAL primitive. Wider at the base (frame edge),
    # narrower at the tip (cell). Real labels @ 512-canvas:
    # base width p50=74 (→ 29 px @200-canvas), tip width p50=59 (→ 23 px),
    # taper (tip/base) p50=0.79 with p10-p90 spread 0.73-0.82.
    p_pipette: float = 0.5
    pipette_base_width_min: float = 20.0
    pipette_base_width_max: float = 34.0
    pipette_taper_min: float = 0.60      # tip / base ratio
    pipette_taper_max: float = 0.90
    # Wall thickness as a fraction of half-width (0 = solid, 1 = paper-thin walls).
    pipette_hollow_frac_min: float = 0.55
    pipette_hollow_frac_max: float = 0.85
    # Interior tint (signed intensity relative to BG). Real pipette interior
    # is neither pure BG nor as dark as the walls — a mild tint either way.
    pipette_interior_amp_min: float = -0.04   # brighter than BG
    pipette_interior_amp_max: float = 0.05    # darker than BG
    # Tip lands at this normalized distance beyond the target cell's edge;
    # 0 = exactly touching, negative = slight penetration.
    pipette_tip_over_cell_min: float = -0.15
    pipette_tip_over_cell_max: float = 0.05
    pipette_amp_min: float = 0.08
    pipette_amp_max: float = 0.25
    pipette_edge_sigma: float = 1.2

    # Cell interior texture. Multi-scale Perlin + a bank of organelle blobs
    # (dark spots scattered inside the cell) — real cells show many discrete
    # organelles, not one uniform nucleus.
    cell_texture_amp_min: float = 0.12
    cell_texture_amp_max: float = 0.32
    cell_texture_steps_coarse: int = 4      # organelle scale
    cell_texture_steps_fine: int = 20       # granular scale
    cell_texture_fine_weight: float = 0.55  # weight on the fine scale
    # Organelles: dark spots scattered inside each cell. Count varies 2-6.
    n_organelles_max: int = 6
    organelle_radius_frac_min: float = 0.05  # of cell radius
    organelle_radius_frac_max: float = 0.30
    organelle_amp_min: float = 0.03
    organelle_amp_max: float = 0.18
    organelle_offset_frac_max: float = 0.70

    # PSF (point spread function). Real brightfield / DIC microscopy has a
    # DoG-like signature — sharp bright core with a distinct dark ring around
    # bright features (from diffraction + phase contrast). We convolve with a
    # DoG kernel = Gauss(core) − w · Gauss(surround), normalized. Earlier
    # weights 0.05-0.35 were too subtle to see; bumped upper bounds.
    psf_core_sigma_min: float = 0.6
    psf_core_sigma_max: float = 2.4
    psf_surround_sigma_min: float = 3.0
    psf_surround_sigma_max: float = 9.0
    psf_surround_weight_min: float = 0.20
    psf_surround_weight_max: float = 0.70

    # ---- Distortions (all applied per-clip, sampled once per video) --------
    # Faintness: multiply foreground signal by scalar in [faint_min, 1]
    faint_min: float = 0.30

    # Defocus: Gaussian blur σ (px). 0 = no blur.
    defocus_sigma_max: float = 2.6

    # Motion blur: box-average over this many consecutive frames. Real cameras
    # typically expose short enough that fast-beating flagella don't smear into
    # multi-position ghosts. Was 4 → smeared into 3-4 ghost flagella.
    motion_blur_max: int = 2

    # Drifting debris blobs. Real debris is subtle — keep count and amp modest.
    n_debris_max: int = 5
    debris_sigma_min: float = 1.0
    debris_sigma_max: float = 3.0
    debris_amp_min: float = 0.005
    debris_amp_max: float = 0.06
    debris_speed_max: float = 5.0           # px per frame

    # STATIC background blobs of varying scales — out-of-focus cells / debris
    # / cytoplasm chunks in real videos. Present at multiple sizes.
    n_bg_blobs_max: int = 14
    bg_blob_sigma_min: float = 2.0
    bg_blob_sigma_max: float = 22.0         # a few "big" blurry blobs
    bg_blob_amp_min: float = 0.010
    bg_blob_amp_max: float = 0.09
    bg_blob_dark_prob: float = 0.55         # P(darker than BG)

    # Vignette: darken corners. Real videos show mild vignetting at most.
    vignette_strength_max: float = 0.10

    # Additive gaussian noise σ (in the same [0,1] intensity units)
    noise_sigma_min: float = 0.005
    noise_sigma_max: float = 0.040

    # Background: procedural mean level in [0, 1]
    bg_level_min: float = 0.40
    bg_level_max: float = 0.65
    bg_texture_amp_max: float = 0.06        # slow 2D perlin-ish texture amplitude


# ---- Helpers --------------------------------------------------------------

def _choice_from_probs(key, probs: Array) -> Array:
    """Categorical draw; returns an integer index."""
    return jax.random.choice(key, probs.shape[0], p=probs)


def _sigmoid_step(d: Array, edge: float, sigma: float) -> Array:
    """Smooth transition 1→0 as d crosses `edge` with softness `sigma`."""
    return 0.5 * (1.0 - jnp.tanh((d - edge) / jnp.maximum(sigma, 1e-3)))


# ---- Cell sampling + contact ----------------------------------------------

def sample_cells(key: jax.Array, cfg: DiverseSimConfig, scene_scale: Array):
    """Sample cell positions/radii/amps and relax overlap via soft-repulsion.

    All spatial extents (radii, drift) are multiplied by `scene_scale`, which
    is a per-clip zoom factor so the sim covers close-up vs far-away views.

    Returns dict with fixed-shape arrays of length `n_max_cells` plus a
    boolean alive mask. Non-alive slots have arbitrary values (masked out
    downstream by their z_pres = 0).
    """
    N = cfg.n_max_cells
    k_n, k_r, k_pos, k_amp, k_drift, k_tex, k_ax, k_ori, k_front = jax.random.split(key, 9)

    n_active = _choice_from_probs(k_n, jnp.array(cfg.n_cells_probs)) + 1
    alive = jnp.arange(N) < n_active

    radii = jax.random.uniform(k_r, (N,), minval=cfg.cell_radius_min, maxval=cfg.cell_radius_max)
    radii = radii * scene_scale
    # Initial positions uniformly inside safe box
    lo = radii + cfg.cell_margin
    hi_y = cfg.H - radii - cfg.cell_margin
    hi_x = cfg.W - radii - cfg.cell_margin
    u = jax.random.uniform(k_pos, (N, 2))
    centers = jnp.stack([lo + u[:, 0] * (hi_y - lo),
                         lo + u[:, 1] * (hi_x - lo)], axis=1)
    amps_mag = jax.random.uniform(k_amp, (N,), minval=cfg.cell_amp_min, maxval=cfg.cell_amp_max)

    # Jacobi soft-repulsion: penalize pairwise overlap only.
    pair_mask = alive[:, None] * alive[None, :] * (1.0 - jnp.eye(N))
    target_dists = radii[:, None] + radii[None, :] + cfg.contact_margin   # (N, N)

    def step(centers, _):
        diff = centers[:, None] - centers[None, :]               # (N, N, 2)
        d = jnp.linalg.norm(diff, axis=-1) + 1e-3                # (N, N)
        overlap = jnp.maximum(target_dists - d, 0.0) * pair_mask  # (N, N)
        force = (overlap / d)[..., None] * diff                  # (N, N, 2)
        centers = centers + cfg.contact_lr * force.sum(axis=1)
        # Keep inside safe box
        min_yx = radii[:, None] + cfg.cell_margin
        centers = jnp.stack([
            jnp.clip(centers[:, 0], min_yx[:, 0], cfg.H - min_yx[:, 0]),
            jnp.clip(centers[:, 1], min_yx[:, 0], cfg.W - min_yx[:, 0]),
        ], axis=1)
        return centers, None

    centers, _ = jax.lax.scan(step, centers, None, length=cfg.contact_iters)

    # Per-cell linear drift: velocity such that total displacement over the
    # clip is ≤ cell_drift_speed_max px in each axis. Random direction.
    dtheta = jax.random.uniform(k_drift, (N,), minval=0.0, maxval=2 * jnp.pi)
    dmag = jax.random.uniform(jax.random.fold_in(k_drift, 1), (N,),
                               minval=0.0, maxval=cfg.cell_drift_speed_max)
    drift = jnp.stack([dmag * jnp.sin(dtheta), dmag * jnp.cos(dtheta)], axis=1)
    tex_amps = jax.random.uniform(k_tex, (N,), minval=cfg.cell_texture_amp_min,
                                    maxval=cfg.cell_texture_amp_max)
    axis_ratio = jax.random.uniform(k_ax, (N,), minval=cfg.cell_axis_ratio_min,
                                     maxval=cfg.cell_axis_ratio_max)
    orientation = jax.random.uniform(k_ori, (N,), minval=0.0, maxval=jnp.pi)
    front_angle = jax.random.uniform(k_front, (N,), minval=0.0,
                                       maxval=2 * jnp.pi)
    # Organelles per cell: n_organelles_max slots, each with (present, offset,
    # radius, amp). Number of active organelles varies per cell.
    n_org = cfg.n_organelles_max
    k_org = jax.random.split(k_tex, 5)
    # 2-N organelles per cell alive (uniform random count)
    n_org_active = jax.random.randint(k_org[0], (N,), 2, n_org + 1)
    org_alive = (jnp.arange(n_org)[None, :] < n_org_active[:, None]
                  ).astype(jnp.float32)                # (N_cells, n_org)
    org_offset = jax.random.uniform(k_org[1], (N, n_org, 2), minval=-1.0,
                                      maxval=1.0) * cfg.organelle_offset_frac_max
    org_rfrac = jax.random.uniform(k_org[2], (N, n_org),
                                     minval=cfg.organelle_radius_frac_min,
                                     maxval=cfg.organelle_radius_frac_max)
    org_amp = jax.random.uniform(k_org[3], (N, n_org),
                                   minval=cfg.organelle_amp_min,
                                   maxval=cfg.organelle_amp_max) * org_alive
    return dict(alive=alive, centers=centers, radii=radii, amps_mag=amps_mag,
                drift=drift, tex_amps=tex_amps, axis_ratio=axis_ratio,
                orientation=orientation, front_angle=front_angle,
                org_offset=org_offset, org_rfrac=org_rfrac, org_amp=org_amp,
                n_active=n_active)


def _cell_soft_mask(cy: Array, cx: Array, radius: Array, axis_ratio: Array,
                    orientation: Array, cfg: DiverseSimConfig) -> Array:
    """Soft mask of one elliptical cell body. major axis = radius; minor axis
    = radius * axis_ratio; oriented by `orientation`."""
    yy, xx = jnp.mgrid[:cfg.H, :cfg.W].astype(jnp.float32)
    dy, dx = yy - cy, xx - cx
    ct, st = jnp.cos(orientation), jnp.sin(orientation)
    # Rotate into ellipse frame (major axis aligned with rotated x)
    x_e = dx * ct + dy * st
    y_e = -dx * st + dy * ct
    # Squash: divide major-axis coord by 1, minor-axis coord by axis_ratio
    d_ellipse = jnp.sqrt(x_e ** 2 + (y_e / jnp.maximum(axis_ratio, 1e-3)) ** 2)
    return _sigmoid_step(d_ellipse, radius, cfg.cell_edge_sigma)


def render_cell_bodies(cells: dict, cfg: DiverseSimConfig, t_norm: Array,
                       texture: Array) -> Array:
    """Sum of smooth elliptical dark bodies (H, W) at normalized time t∈[0,1].

    Cells drift linearly: center(t) = center(0) + drift · (t_norm − 0.5) so
    the middle frame is the sampled position. `texture` is a shared (H, W)
    Perlin field (multi-scale sum) that adds interior variation inside the
    soft mask. Each cell additionally gets an optional dark nucleus spot."""
    yy, xx = jnp.mgrid[:cfg.H, :cfg.W].astype(jnp.float32)
    offset = cells["drift"] * (t_norm - 0.5)                      # (N, 2)
    def one(center0, off, radius, ratio, orient, amp_mag, tex_amp,
             org_offset, org_rfrac, org_amp, alive):
        cy, cx = center0[0] + off[0], center0[1] + off[1]
        soft = _cell_soft_mask(cy, cx, radius, ratio, orient, cfg)
        base = -amp_mag * soft
        interior = -tex_amp * texture * soft
        # Organelles: multiple soft dark spots inside the cell, confined by
        # the outer soft mask. Rotate offsets by orient so they stay inside
        # the ellipse.
        ct, st = jnp.cos(orient), jnp.sin(orient)
        def one_org(off_i, rfrac_i, amp_i):
            oy_local = off_i[0] * radius * ratio
            ox_local = off_i[1] * radius
            oy = cy + ox_local * st + oy_local * ct
            ox = cx + ox_local * ct - oy_local * st
            r = rfrac_i * radius
            d2 = (yy - oy) ** 2 + (xx - ox) ** 2
            return -amp_i * jnp.exp(-0.5 * d2 / jnp.maximum(r ** 2, 1.0))
        orgs = jax.vmap(one_org)(org_offset, org_rfrac, org_amp).sum(axis=0)
        organelles = orgs * soft                              # confine
        return (base + interior + organelles) * alive.astype(jnp.float32)
    contribs = jax.vmap(one)(cells["centers"], offset, cells["radii"],
                              cells["axis_ratio"], cells["orientation"],
                              cells["amps_mag"], cells["tex_amps"],
                              cells["org_offset"], cells["org_rfrac"],
                              cells["org_amp"], cells["alive"])
    return contribs.sum(axis=0)


def cells_interior_mask(cells: dict, cfg: DiverseSimConfig, t_norm: Array) -> Array:
    """Union of alive cell interiors (H, W) in [0, 1]. Used to mask out any
    flagellum contribution that falls inside a cell body."""
    offset = cells["drift"] * (t_norm - 0.5)
    def one(center0, off, radius, ratio, orient, alive):
        cy, cx = center0[0] + off[0], center0[1] + off[1]
        return _cell_soft_mask(cy, cx, radius, ratio, orient, cfg) * alive.astype(jnp.float32)
    masks = jax.vmap(one)(cells["centers"], offset, cells["radii"],
                            cells["axis_ratio"], cells["orientation"],
                            cells["alive"])
    # union via clip on the sum
    return jnp.clip(masks.sum(axis=0), 0.0, 1.0)


# ---- Pipette --------------------------------------------------------------

def sample_pipette(key: jax.Array, cfg: DiverseSimConfig, cells: dict,
                   scene_scale: Array) -> dict:
    """One (possibly-absent) CONICAL pipette per clip. Its tip lands on a
    target cell's membrane (real pipettes ALWAYS touch the cell they're
    holding). Enters from the nearest frame edge along a straight axis."""
    kk = jax.random.split(key, 8)
    present = jax.random.uniform(kk[0], ()) < cfg.p_pipette

    alive = cells["alive"].astype(jnp.float32)
    logits = alive - 1e6 * (1.0 - alive)
    tgt = jax.random.categorical(kk[1], logits)
    target_center = cells["centers"][tgt]
    target_radius = cells["radii"][tgt]

    base_width = jax.random.uniform(kk[2], (),
                                     minval=cfg.pipette_base_width_min,
                                     maxval=cfg.pipette_base_width_max) * scene_scale
    taper = jax.random.uniform(kk[3], (), minval=cfg.pipette_taper_min,
                                maxval=cfg.pipette_taper_max)
    tip_width = base_width * taper
    amp = jax.random.uniform(kk[4], (), minval=cfg.pipette_amp_min,
                              maxval=cfg.pipette_amp_max)
    interior_amp = jax.random.uniform(kk[5], (),
                                        minval=cfg.pipette_interior_amp_min,
                                        maxval=cfg.pipette_interior_amp_max)

    approach = jax.random.uniform(kk[6], (), minval=0.0, maxval=2 * jnp.pi)
    ay = jnp.sin(approach)
    ax = jnp.cos(approach)

    over = jax.random.uniform(kk[7], (), minval=cfg.pipette_tip_over_cell_min,
                                maxval=cfg.pipette_tip_over_cell_max)
    tip_offset = target_radius * (1.0 - over)
    tip = jnp.stack([target_center[0] - ay * tip_offset,
                     target_center[1] - ax * tip_offset])

    base_offset = float(max(cfg.H, cfg.W))
    base = jnp.stack([tip[0] - ay * base_offset,
                      tip[1] - ax * base_offset])

    drift = jnp.zeros((2,), jnp.float32)

    hollow_frac = jax.random.uniform(jax.random.fold_in(kk[7], 42), (),
                                       minval=cfg.pipette_hollow_frac_min,
                                       maxval=cfg.pipette_hollow_frac_max)

    return dict(present=present, tip=tip, base=base,
                base_width=base_width, tip_width=tip_width,
                amp=amp, interior_amp=interior_amp,
                hollow_frac=hollow_frac, drift=drift, target_cell=tgt)


def render_pipette(pip: dict, cfg: DiverseSimConfig, t_norm: Array) -> Array:
    """Render one HOLLOW CONICAL pipette. Outer soft mask with axial width
    linearly interpolated (base → tip), minus an inner soft mask (hollow), plus
    an interior tint (signed) modulated by the inner mask."""
    yy, xx = jnp.mgrid[:cfg.H, :cfg.W].astype(jnp.float32)
    off = pip["drift"] * (t_norm - 0.5)
    p0 = pip["base"] + off       # wider end (frame edge)
    p1 = pip["tip"] + off        # narrower end (touches cell)

    axis = p1 - p0
    axis_len2 = jnp.sum(axis ** 2) + 1e-6
    d_y = yy - p0[0]; d_x = xx - p0[1]
    t = jnp.clip((d_y * axis[0] + d_x * axis[1]) / axis_len2, 0.0, 1.0)
    proj_y = p0[0] + t * axis[0]
    proj_x = p0[1] + t * axis[1]
    perp_d = jnp.sqrt((yy - proj_y) ** 2 + (xx - proj_x) ** 2)

    # Width along the axis: linearly interpolate base → tip
    outer_r = (pip["base_width"] + t * (pip["tip_width"] - pip["base_width"])) * 0.5
    inner_r = outer_r * pip["hollow_frac"]
    outer = _sigmoid_step(perp_d, outer_r, cfg.pipette_edge_sigma)
    inner = _sigmoid_step(perp_d, inner_r, cfg.pipette_edge_sigma)
    walls = jnp.maximum(outer - inner, 0.0)
    # Contribution: dark walls (−amp) + interior tint (signed) on the inner mask.
    signal = -pip["amp"] * walls + (-pip["interior_amp"]) * inner
    return signal * pip["present"].astype(jnp.float32)


# ---- Flagellum sampling ---------------------------------------------------

def sample_flagella(key: jax.Array, cfg: DiverseSimConfig, cells: dict,
                    scene_scale: Array) -> dict:
    """Paired-symmetric flagellum sampler.

    Structural change from earlier version: slots are laid out as PAIRS
    (0,1), (2,3), ...  Pair p belongs to cell p. Slot 2p is the "primary"
    flagellum and 2p+1 its mirror. When a cell has 1 flagellum only slot 2p
    is alive; when it has 2, both are alive and share beat params, with
    attachment angles symmetric around the cell's `front_angle`.

    All lengths / widths scale by `scene_scale`.
    """
    assert cfg.n_max_flagella == 2 * cfg.n_max_cells, (
        "paired layout: n_max_flagella must be 2 * n_max_cells")
    Nc = cfg.n_max_cells
    N = cfg.n_max_flagella
    kk = jax.random.split(key, 12)

    # Per-cell number of flagella (0..len(n_flag_per_cell_probs)-1, cap at 2)
    n_per_cell = jax.random.choice(
        kk[0], jnp.arange(len(cfg.n_flag_per_cell_probs)),
        shape=(Nc,), p=jnp.array(cfg.n_flag_per_cell_probs))
    n_per_cell = jnp.minimum(n_per_cell, 2) * cells["alive"].astype(jnp.int32)

    # Slot indexing
    idx = jnp.arange(N)
    slot_cell = idx // 2                          # (N,)  0,0,1,1,...
    slot_in_pair = idx % 2                        # (N,)  0,1,0,1,...
    alive = slot_in_pair < n_per_cell[slot_cell]
    parent = jnp.where(alive, slot_cell, -1)      # -1 for dead slots

    # Per-cell "shared" parameters (beat params + pair spread) — one draw per cell.
    kc = jax.random.split(kk[1], 12)
    cell_curv_amp = jax.random.uniform(kc[0], (Nc,), minval=cfg.curv_amp_min,
                                         maxval=cfg.curv_amp_max)
    cell_curv_k = jax.random.uniform(kc[1], (Nc,), minval=cfg.curv_k_min,
                                       maxval=cfg.curv_k_max)
    cell_curv_freq = jax.random.uniform(kc[2], (Nc,), minval=cfg.curv_freq_min,
                                          maxval=cfg.curv_freq_max)
    cell_curv_phi = jax.random.uniform(kc[3], (Nc,), minval=0.0,
                                         maxval=2 * jnp.pi)
    cell_tip_alpha = jax.random.uniform(kc[4], (Nc,), minval=cfg.tip_alpha_min,
                                          maxval=cfg.tip_alpha_max)
    cell_breast_h2 = jax.random.uniform(kc[5], (Nc,), minval=cfg.breast_h2_min,
                                          maxval=cfg.breast_h2_max)
    cell_breast_dphi = jax.random.uniform(kc[6], (Nc,), minval=0.0,
                                            maxval=2 * jnp.pi)
    cell_static_amp = jax.random.uniform(kc[7], (Nc,), minval=cfg.static_bend_min,
                                           maxval=cfg.static_bend_max)
    cell_static_phase = jax.random.uniform(kc[8], (Nc,), minval=0.0,
                                             maxval=2 * jnp.pi)
    cell_lat_amp = jax.random.uniform(kc[9], (Nc,), minval=cfg.lat_amp_min,
                                        maxval=cfg.lat_amp_max)
    cell_lat_wave = jax.random.uniform(kc[10], (Nc,), minval=cfg.lat_wave_min,
                                         maxval=cfg.lat_wave_max)
    cell_lat_freq = jax.random.uniform(kc[11], (Nc,), minval=cfg.lat_freq_min,
                                         maxval=cfg.lat_freq_max)
    kc2 = jax.random.split(kk[2], 4)
    cell_lat_phi = jax.random.uniform(kc2[0], (Nc,), minval=0.0,
                                        maxval=2 * jnp.pi)
    mode_probs = jnp.array(cfg.mode_weights) / sum(cfg.mode_weights)
    cell_mode = jax.random.choice(kc2[1], N_MODES, shape=(Nc,), p=mode_probs)
    # Half-angular separation between the two flagella of a pair. Was π/8-π/3;
    # π/8 (22.5°) makes them visually merge — bump min to π/5 (36°) so they're
    # always distinguishable.
    cell_pair_spread = jax.random.uniform(kc2[2], (Nc,),
                                            minval=jnp.pi / 5,
                                            maxval=jnp.pi / 2)
    # Biased amp sampling: u^bias skews toward the low end so most flagella
    # are faint (low contrast against BG / cell body). bias=1 uniform,
    # bias=2.5 → median ≈ 0.14 vs uniform 0.36.
    u_amp = jax.random.uniform(kc2[3], (Nc,))
    cell_amp_mag = cfg.flag_amp_min + (cfg.flag_amp_max - cfg.flag_amp_min) * \
                     u_amp ** cfg.flag_amp_bias

    kc3 = jax.random.split(kk[3], 4)
    cell_length = jax.random.uniform(kc3[0], (Nc,), minval=cfg.flag_length_min,
                                       maxval=cfg.flag_length_max) * scene_scale
    cell_width = jax.random.uniform(kc3[1], (Nc,), minval=cfg.flag_width_min,
                                      maxval=cfg.flag_width_max) * scene_scale
    cell_sign = jnp.where(jax.random.uniform(kc3[2], (Nc,)) < cfg.flag_dark_prob,
                            -1.0, 1.0)
    cell_wave_sign = jnp.where(jax.random.uniform(kc3[3], (Nc,)) < 0.5, 1.0, -1.0)

    # Gather per-slot from per-cell (slots in the same pair share these).
    curv_amp = cell_curv_amp[slot_cell]
    curv_k = cell_curv_k[slot_cell]
    curv_freq = cell_curv_freq[slot_cell]
    curv_phi = cell_curv_phi[slot_cell]
    tip_alpha = cell_tip_alpha[slot_cell]
    breast_h2 = cell_breast_h2[slot_cell]
    breast_dphi = cell_breast_dphi[slot_cell]
    static_amp = cell_static_amp[slot_cell]
    static_phase = cell_static_phase[slot_cell]
    lat_amp = cell_lat_amp[slot_cell]
    lat_wave = cell_lat_wave[slot_cell]
    lat_freq = cell_lat_freq[slot_cell]
    lat_phi = cell_lat_phi[slot_cell]
    mode = cell_mode[slot_cell]
    length = cell_length[slot_cell]
    width = cell_width[slot_cell]
    amp_mag = cell_amp_mag[slot_cell]
    sign = cell_sign[slot_cell]
    wave_sign = cell_wave_sign[slot_cell]

    # Attachment angle: symmetric about front, +spread for one slot, −spread
    # for the other. If singleton (only slot 0 alive), attach at front.
    front = cells["front_angle"][slot_cell]
    spread = cell_pair_spread[slot_cell]
    # slot 0 → -1, slot 1 → +1
    signdir = 2.0 * slot_in_pair.astype(jnp.float32) - 1.0
    singleton = (n_per_cell[slot_cell] == 1)
    theta_attach = front + jnp.where(singleton, 0.0, signdir * spread)

    # Attachment on the elliptical membrane: use the ellipse's local geometry.
    # For simplicity anchor at r_effective * (sin, cos θ) — for elongated
    # cells this slightly under-anchors on the minor axis; good enough.
    parent_safe = jnp.clip(slot_cell, 0, Nc - 1)
    pc_center = cells["centers"][parent_safe]
    pc_radius = cells["radii"][parent_safe]
    pc_ratio = cells["axis_ratio"][parent_safe]
    pc_orient = cells["orientation"][parent_safe]
    # Compute point on ellipse rim at world angle theta_attach:
    # parametrise in ellipse frame using an offset from theta_attach relative
    # to orient. Local coord: (cos α, sin α * ratio) * radius (major aligned x_e).
    alpha = theta_attach - pc_orient
    ex = pc_radius * jnp.cos(alpha)
    ey = pc_radius * pc_ratio * jnp.sin(alpha)
    # Rotate back to world
    ct = jnp.cos(pc_orient); st = jnp.sin(pc_orient)
    attach_y = pc_center[:, 0] + (ex * st + ey * ct)
    attach_x = pc_center[:, 1] + (ex * ct - ey * st)
    attachment = jnp.stack([attach_y, attach_x], axis=1)

    # Base angle: pointing radially outward from cell center (approximately).
    base_angle = jnp.arctan2(attach_y - pc_center[:, 0],
                              attach_x - pc_center[:, 1])
    # Flip breast/lat phase for the mirror partner so their beats are mirror-
    # image rather than in-phase — matches Chlamydomonas breaststroke asymmetry.
    curv_phi = jnp.where((slot_in_pair == 1) & (mode == MODE_BREASTSTROKE),
                          curv_phi + jnp.pi, curv_phi)

    amp = amp_mag * sign

    return dict(
        alive=alive, parent=parent, is_attached=alive,
        attachment=attachment, base_angle=base_angle, length=length,
        width=width, amp=amp, mode=mode,
        curv_amp=curv_amp, curv_k=curv_k, curv_freq=curv_freq, curv_phi=curv_phi,
        tip_alpha=tip_alpha,
        breast_h2=breast_h2, breast_dphi=breast_dphi,
        static_amp=static_amp, static_phase=static_phase,
        lat_amp=lat_amp, lat_wave=lat_wave, lat_freq=lat_freq, lat_phi=lat_phi,
        wave_sign=wave_sign,
    )


# ---- Flagellum curve construction -----------------------------------------

def _curve_from_curvature(kappa: Array, base_pos: Array, base_angle: Array,
                          ds_px: Array, ds_s: Array) -> Array:
    """Integrate `κ(s)` to a planar curve starting at `base_pos` with tangent
    `base_angle`.

    Two distinct step sizes:
      ds_s   — s-space step (unitless, 1/(K-1)); used for angle integration
               so `curv_amp` has units of radians / s-unit (independent of
               the flagellum's pixel length).
      ds_px  — pixel step (length_px / (K-1)); used for spatial (y, x) walk.

    Args:
      kappa:      (K,) curvature per s-unit (radians / unitless-s-step).
      base_pos:   (2,) starting (y, x) position.
      base_angle: scalar starting tangent angle (radians).
      ds_px:      (K,) or scalar pixel step.
      ds_s:       (K,) or scalar s-step.

    Returns:
      (K, 2) polyline in (y, x).
    """
    theta = base_angle + jnp.cumsum(kappa * ds_s)
    dy = jnp.sin(theta) * ds_px
    dx = jnp.cos(theta) * ds_px
    y = base_pos[0] + jnp.cumsum(dy)
    x = base_pos[1] + jnp.cumsum(dx)
    return jnp.stack([y, x], axis=-1)


def flagellum_curve_at_t(params_i: dict, t_norm: Array, cfg: DiverseSimConfig) -> Array:
    """Compute one flagellum's polyline (K, 2) at normalized time `t_norm`.

    Selects among five parametric modes and blends by a one-hot on `mode`.
    """
    K = cfg.n_arc_pts
    s = jnp.linspace(0.0, 1.0, K)                            # normalized arc length
    ds_px = jnp.ones((K,), jnp.float32) * (params_i["length"] / (K - 1))
    ds_s = jnp.ones((K,), jnp.float32) * (1.0 / (K - 1))

    two_pi = 2.0 * jnp.pi
    ws = params_i["wave_sign"]

    # 1) curvature wave (uniform amplitude)
    kappa_curv = params_i["curv_amp"] * jnp.sin(
        two_pi * (params_i["curv_k"] * s + ws * params_i["curv_freq"] * t_norm)
        + params_i["curv_phi"])

    # 2) tip-amplified curvature wave
    kappa_tip = (params_i["curv_amp"] * s ** params_i["tip_alpha"]) * jnp.sin(
        two_pi * (params_i["curv_k"] * s + ws * params_i["curv_freq"] * t_norm)
        + params_i["curv_phi"])

    # 3) breaststroke: first + second harmonic
    phase1 = two_pi * (params_i["curv_k"] * s + ws * params_i["curv_freq"] * t_norm) \
             + params_i["curv_phi"]
    kappa_breast = params_i["curv_amp"] * (jnp.sin(phase1)
                                            + params_i["breast_h2"] *
                                            jnp.sin(2.0 * phase1 + params_i["breast_dphi"]))

    # 4) near-static: fixed κ(s) shape + tiny time flicker
    static_shape = params_i["static_amp"] * jnp.sin(
        two_pi * (params_i["curv_k"] * s) + params_i["static_phase"])
    flicker = cfg.static_flicker * jnp.sin(two_pi * t_norm + params_i["static_phase"])
    kappa_static = static_shape * (1.0 + flicker)

    # 5) lateral sine: no curvature integration — just lateral displacement of
    #    a straight rod. Handled separately below.
    lat_disp = params_i["lat_amp"] * jnp.sin(
        two_pi * (params_i["lat_wave"] * s + ws * params_i["lat_freq"] * t_norm)
        + params_i["lat_phi"])
    # Straight-rod backbone in the local frame:
    s_len = s * params_i["length"]                               # (K,)
    lat_local = jnp.stack([s_len, lat_disp], axis=-1)            # (K, 2) = (along, lateral)
    ba = params_i["base_angle"]
    R = jnp.array([[jnp.sin(ba), jnp.cos(ba)], [jnp.cos(ba), -jnp.sin(ba)]])
    # ↑ maps (along, lateral) → (y, x): y = sin·along + cos·lateral,
    #                                   x = cos·along − sin·lateral
    curve_lat = params_i["attachment"] + lat_local @ R.T

    # Curves for modes 0..3 come from κ-integration
    def integrate(kappa):
        return _curve_from_curvature(kappa, params_i["attachment"],
                                     params_i["base_angle"], ds_px, ds_s)

    curve_curv = integrate(kappa_curv)
    curve_tip = integrate(kappa_tip)
    curve_breast = integrate(kappa_breast)
    curve_static = integrate(kappa_static)

    # Stack and select by mode
    curves = jnp.stack([curve_curv, curve_tip, curve_breast, curve_static, curve_lat],
                       axis=0)                                  # (N_MODES, K, 2)
    onehot = jax.nn.one_hot(params_i["mode"], N_MODES)          # (N_MODES,)
    return (onehot[:, None, None] * curves).sum(axis=0)          # (K, 2)


# ---- Rasterization --------------------------------------------------------

def render_flagellum_frame(curve_yx: Array, width_px: Array, amp: Array,
                           alive: Array, cfg: DiverseSimConfig) -> Array:
    """Render one flagellum's signed contribution (H, W) at one frame.

    Uses the existing SDF renderer, which expects points in [-1, 1] normalized
    coords and returns a Gaussian-around-polyline mask in [0, 1].
    """
    # Convert (y, x) pixel coords to (x, y) normalized ∈ [-1, 1]
    xy_norm = jnp.stack([
        curve_yx[:, 1] / (cfg.W - 1) * 2.0 - 1.0,
        curve_yx[:, 0] / (cfg.H - 1) * 2.0 - 1.0,
    ], axis=-1)
    # NOTE: render_polyline_segments_sdf assumes a square canvas of side `res`.
    # Our canvas is H×W (may be non-square); we render at max side and crop.
    res = max(cfg.H, cfg.W)
    mask = render_polyline_segments_sdf(xy_norm, res, sigma=width_px)
    # Crop to (H, W) — center align
    y0 = (res - cfg.H) // 2
    x0 = (res - cfg.W) // 2
    mask = jax.lax.dynamic_slice(mask, (y0, x0), (cfg.H, cfg.W))
    return amp * mask * alive.astype(jnp.float32)


# ---- Distortions ----------------------------------------------------------

def _gaussian_kernel_1d(sigma: Array, radius: int) -> Array:
    x = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    k = jnp.exp(-0.5 * (x / jnp.maximum(sigma, 1e-3)) ** 2)
    return k / k.sum()


def gaussian_blur_2d(frame: Array, sigma: Array, radius: int = 8) -> Array:
    """Separable 2D Gaussian blur with REFLECT padding at borders. Zero-pad
    (the default in `jax.scipy.signal.convolve(mode='same')`) darkens the
    frame edges — we pad with reflection so brightness is preserved."""
    k1d = _gaussian_kernel_1d(sigma, radius)
    padded = jnp.pad(frame, ((radius, radius), (radius, radius)), mode="reflect")
    blurred = jax.scipy.signal.convolve(padded, k1d[None, :], mode="same")
    blurred = jax.scipy.signal.convolve(blurred, k1d[:, None], mode="same")
    return blurred[radius:-radius, radius:-radius]


def apply_defocus(clip: Array, sigma: Array) -> Array:
    """Per-frame Gaussian blur; sigma is a scalar (same across frames)."""
    return jax.vmap(lambda f: gaussian_blur_2d(f, sigma))(clip)


def apply_psf(clip: Array, core_sigma: Array, surround_sigma: Array,
              surround_weight: Array) -> Array:
    """DoG-style PSF: sharp core − weighted wider surround, then normalized
    to preserve DC. Approximates the diffraction-limited bright-core-with-
    dark-halo pattern real brightfield / DIC microscopy shows around edges.

    Equivalent to convolving with kernel = Gauss(core_σ) − w·Gauss(surround_σ),
    then dividing by (1 − w) so a flat image is preserved."""
    def one(f):
        core = gaussian_blur_2d(f, core_sigma)
        surround = gaussian_blur_2d(f, surround_sigma)
        return (core - surround_weight * surround) / jnp.maximum(
            1.0 - surround_weight, 1e-3)
    return jax.vmap(one)(clip)


def add_bg_blobs(key: jax.Array, cfg: DiverseSimConfig) -> Array:
    """Static (non-drifting) background blobs of varying scales — out-of-focus
    debris, cytoplasm chunks. Returns (H, W) additive."""
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    N = cfg.n_bg_blobs_max
    n_active = jax.random.randint(k1, (), 0, N + 1)
    alive = (jnp.arange(N) < n_active).astype(jnp.float32)
    cy = jax.random.uniform(k2, (N,), minval=0.0, maxval=cfg.H)
    cx = jax.random.uniform(k3, (N,), minval=0.0, maxval=cfg.W)
    # Log-uniform sigma so we get many small + few large blobs
    lo = jnp.log(cfg.bg_blob_sigma_min); hi = jnp.log(cfg.bg_blob_sigma_max)
    sigma = jnp.exp(jax.random.uniform(k4, (N,), minval=lo, maxval=hi))
    mag = jax.random.uniform(k5, (N,), minval=cfg.bg_blob_amp_min,
                              maxval=cfg.bg_blob_amp_max)
    sign = jnp.where(jax.random.uniform(k6, (N,)) < cfg.bg_blob_dark_prob,
                      -1.0, 1.0)
    amp = mag * sign * alive
    yy, xx = jnp.mgrid[:cfg.H, :cfg.W].astype(jnp.float32)
    def one(cy_i, cx_i, s_i, a_i):
        d2 = (yy - cy_i) ** 2 + (xx - cx_i) ** 2
        return a_i * jnp.exp(-0.5 * d2 / (s_i ** 2 + 1e-3))
    return jax.vmap(one)(cy, cx, sigma, amp).sum(axis=0)


def apply_motion_blur(clip: Array, k: int) -> Array:
    """Box-average over k consecutive frames (temporal, causal-symmetric).

    k is a runtime int chosen from 1..motion_blur_max at trace-time. We
    implement it as a fixed-size conv with a mask that zeros out the tail
    when the runtime k is smaller than the compile-time maximum.
    """
    return clip  # We implement in main path with jit-friendly branching below.


def _temporal_box_blur(clip: Array, weights: Array) -> Array:
    """Temporal 1D conv with `weights` (odd length). clip: (T, H, W)."""
    T = clip.shape[0]
    r = weights.shape[0] // 2
    pad = [(r, r), (0, 0), (0, 0)]
    clip_p = jnp.pad(clip, pad, mode="edge")
    # Manual convolution over T
    stacked = jnp.stack(
        [clip_p[t : t + T] for t in range(weights.shape[0])], axis=0)  # (W, T, H, W)
    return (weights[:, None, None, None] * stacked).sum(axis=0)


def add_debris(key: jax.Array, cfg: DiverseSimConfig) -> Array:
    """Independent-of-scene drifting blob layer, returns (T, H, W) additive."""
    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)
    N = cfg.n_debris_max
    n_active = jax.random.randint(k1, (), 0, N + 1)
    alive = (jnp.arange(N) < n_active).astype(jnp.float32)

    y0 = jax.random.uniform(k2, (N,), minval=0.0, maxval=cfg.H)
    x0 = jax.random.uniform(k3, (N,), minval=0.0, maxval=cfg.W)
    vy = jax.random.uniform(k4, (N,), minval=-cfg.debris_speed_max, maxval=cfg.debris_speed_max)
    vx = jax.random.uniform(k5, (N,), minval=-cfg.debris_speed_max, maxval=cfg.debris_speed_max)
    sigma = jax.random.uniform(k6, (N,), minval=cfg.debris_sigma_min,
                                maxval=cfg.debris_sigma_max)
    mag = jax.random.uniform(k7, (N,), minval=cfg.debris_amp_min,
                              maxval=cfg.debris_amp_max)
    sign = jnp.where(jax.random.uniform(k8, (N,)) < 0.7, -1.0, 1.0)
    amp = mag * sign * alive

    yy, xx = jnp.mgrid[:cfg.H, :cfg.W].astype(jnp.float32)

    def one_debris(t):
        cy = y0 + vy * t
        cx = x0 + vx * t
        def one(cy_i, cx_i, s_i, a_i):
            d2 = (yy - cy_i) ** 2 + (xx - cx_i) ** 2
            return a_i * jnp.exp(-0.5 * d2 / (s_i ** 2 + 1e-3))
        return jax.vmap(one)(cy, cx, sigma, amp).sum(axis=0)

    return jax.vmap(one_debris)(jnp.arange(cfg.T).astype(jnp.float32))


def vignette(cfg: DiverseSimConfig, strength: Array) -> Array:
    """Radial (1 - strength · r²/r_max²) darkening, shape (H, W)."""
    yy, xx = jnp.mgrid[:cfg.H, :cfg.W].astype(jnp.float32)
    cy, cx = cfg.H / 2.0, cfg.W / 2.0
    r2 = ((yy - cy) ** 2 + (xx - cx) ** 2)
    r2max = (cy ** 2 + cx ** 2)
    return 1.0 - strength * (r2 / r2max)


def slow_bg_texture(key: jax.Array, cfg: DiverseSimConfig, amp: Array) -> Array:
    """Very-low-frequency sinusoidal 2D texture, (H, W). Mimics uneven illum."""
    k1, k2, k3 = jax.random.split(key, 3)
    kx = jax.random.uniform(k1, (), minval=0.3, maxval=1.5)
    ky = jax.random.uniform(k2, (), minval=0.3, maxval=1.5)
    phi = jax.random.uniform(k3, (), minval=0.0, maxval=2 * jnp.pi)
    yy, xx = jnp.mgrid[:cfg.H, :cfg.W].astype(jnp.float32)
    u = yy / cfg.H * 2 * jnp.pi
    v = xx / cfg.W * 2 * jnp.pi
    return amp * jnp.sin(ky * u + kx * v + phi)


# ---- Full pipeline --------------------------------------------------------

@partial(jax.jit, static_argnames=("cfg",))
def sample_clip(key: jax.Array, cfg: DiverseSimConfig) -> dict:
    """Sample one full diverse clip.

    Returns dict with:
      clip_raw     — (T, H, W) full composite in [0, 1] intensity units
      clip_median  — (T, H, W) `clip_raw - global temporal median`
      cells        — cell param dict
      flagella     — flagellum param dict (rest-shape latents)
      curves       — (T, N, K, 2) flagellum curves in pixel space
      bg_level     — scalar
    """
    k_scene, k_bg, k_dist, k_debris, k_noise, k_wave = jax.random.split(key, 6)
    k_cells, k_flag, k_pip, k_ctex, k_scale = jax.random.split(k_scene, 5)

    # Per-clip zoom factor multiplies cell radius + flagellum length/width
    # + pipette dims (but NOT noise/drift budgets).
    scene_scale = jax.random.uniform(k_scale, (), minval=cfg.scene_scale_min,
                                       maxval=cfg.scene_scale_max)

    cells = sample_cells(k_cells, cfg, scene_scale)
    flag = sample_flagella(k_flag, cfg, cells, scene_scale)
    pip = sample_pipette(k_pip, cfg, cells, scene_scale)
    # Multi-scale Perlin cell interior texture (organelle + granular scales).
    k_ctex_c, k_ctex_f = jax.random.split(k_ctex)
    res = max(cfg.H, cfg.W)
    coarse = perlin_noise(k_ctex_c, res, cfg.cell_texture_steps_coarse)
    fine = perlin_noise(k_ctex_f, res, cfg.cell_texture_steps_fine)
    cell_texture = ((1.0 - cfg.cell_texture_fine_weight) * coarse
                     + cfg.cell_texture_fine_weight * fine)
    cell_texture = cell_texture[:cfg.H, :cfg.W]

    # Distortion params (sampled once per video)
    kd = jax.random.split(k_dist, 9)
    faint = jax.random.uniform(kd[0], (), minval=cfg.faint_min, maxval=1.0)
    defocus_sigma = jax.random.uniform(kd[1], (), minval=0.0,
                                        maxval=cfg.defocus_sigma_max)
    motion_k = jax.random.randint(kd[2], (), 1, cfg.motion_blur_max + 1)
    vig_strength = jax.random.uniform(kd[3], (), minval=0.0,
                                       maxval=cfg.vignette_strength_max)
    noise_sigma = jax.random.uniform(kd[4], (), minval=cfg.noise_sigma_min,
                                      maxval=cfg.noise_sigma_max)
    bg_level = jax.random.uniform(kd[5], (), minval=cfg.bg_level_min,
                                   maxval=cfg.bg_level_max)
    # DoG PSF params: sharp core − weighted wider surround.
    psf_core_sigma = jax.random.uniform(kd[6], (),
                                         minval=cfg.psf_core_sigma_min,
                                         maxval=cfg.psf_core_sigma_max)
    psf_surround_sigma = jax.random.uniform(kd[7], (),
                                             minval=cfg.psf_surround_sigma_min,
                                             maxval=cfg.psf_surround_sigma_max)
    psf_surround_weight = jax.random.uniform(
        jax.random.fold_in(kd[7], 1), (),
        minval=cfg.psf_surround_weight_min,
        maxval=cfg.psf_surround_weight_max)
    k_bg_blobs = kd[8]
    bg_tex_amp = jax.random.uniform(k_bg, (), minval=0.0,
                                     maxval=cfg.bg_texture_amp_max)
    bg_tex = slow_bg_texture(k_bg, cfg, bg_tex_amp)                          # (H, W)

    # ---- Compute per-frame flagellum curves + rasterize ------------------
    t_axis = jnp.arange(cfg.T, dtype=jnp.float32) / max(cfg.T - 1, 1)

    def curves_at_t(t_norm):
        # (n_max_flag, K, 2)
        def per(idx):
            params_i = {k: v[idx] if hasattr(v, "shape") and v.shape[:1] == (cfg.n_max_flagella,) else v
                        for k, v in flag.items()}
            return flagellum_curve_at_t(params_i, t_norm, cfg)
        return jax.vmap(per)(jnp.arange(cfg.n_max_flagella))

    curves = jax.vmap(curves_at_t)(t_axis)                                    # (T, N, K, 2)

    def render_t(curves_t):
        def one(curve, w, a, al):
            return render_flagellum_frame(curve, w, a, al, cfg)
        # (N, H, W); sum over slots
        return jax.vmap(one)(curves_t, flag["width"], flag["amp"], flag["alive"]).sum(0)

    flag_layer_t = jax.vmap(render_t)(curves)                                 # (T, H, W)

    # ---- Per-frame cell + pipette layers (drift over the clip) -----------
    cell_layer_t = jax.vmap(lambda tn: render_cell_bodies(cells, cfg, tn,
                                                           cell_texture))(t_axis)
    pip_layer_t = jax.vmap(lambda tn: render_pipette(pip, cfg, tn))(t_axis)

    # Mask flagella + pipette INSIDE any cell body — cells occlude both.
    cell_interior_t = jax.vmap(lambda tn: cells_interior_mask(cells, cfg, tn))(t_axis)
    flag_layer_t = flag_layer_t * (1.0 - cell_interior_t)
    pip_layer_t = pip_layer_t * (1.0 - cell_interior_t)

    # ---- Compose: BG level + slow texture + BG blobs + cell + pipette + flag
    # Cells + flagella + pipette are signed additive contributions; scale
    # flagellum by faint. Pipette/cells inherit their own amplitudes.
    bg_blobs = add_bg_blobs(k_bg_blobs, cfg)                                   # (H, W)
    fg_signed = cell_layer_t + pip_layer_t + faint * flag_layer_t             # (T, H, W)
    clip = bg_level + bg_tex[None] + bg_blobs[None] + fg_signed               # (T, H, W)

    # ---- Debris (independent moving distractors) --------------------------
    debris_layer = add_debris(k_debris, cfg)                                  # (T, H, W)
    clip = clip + debris_layer

    # ---- Vignette (multiplicative, per-frame identical) -------------------
    v = vignette(cfg, vig_strength)                                            # (H, W)
    clip = clip * v[None]

    # ---- PSF (DoG: core − w · surround) ---------------------------------
    clip = apply_psf(clip, psf_core_sigma, psf_surround_sigma, psf_surround_weight)

    # ---- Motion blur: temporal box average with mask matching runtime k ---
    max_k = cfg.motion_blur_max
    # box weights of size max_k, first motion_k entries = 1/motion_k, rest = 0
    idx = jnp.arange(max_k)
    weights = jnp.where(idx < motion_k, 1.0 / motion_k.astype(jnp.float32), 0.0)
    # Convert into odd-length symmetric weights of size 2*max_k-1: centered box
    # of size motion_k over the center. Simpler: pre-build 2*max_k-1 length
    # weights symmetrically indexed around 0.
    center = max_k - 1
    off = jnp.arange(2 * max_k - 1) - center                                   # (2*max_k-1,)
    in_box = jnp.abs(off) < motion_k
    weights_sym = jnp.where(in_box, 1.0, 0.0)
    weights_sym = weights_sym / jnp.maximum(weights_sym.sum(), 1e-3)
    clip = _temporal_box_blur(clip, weights_sym)

    # ---- Additive noise ---------------------------------------------------
    noise = jax.random.normal(k_noise, clip.shape) * noise_sigma
    clip = clip + noise

    # Clip to a wider range than [0,1] (motion blur can push outside);
    # keep sim faithful to real cameras that saturate at 1 — clip conservatively.
    clip = jnp.clip(clip, 0.0, 1.0)

    # ---- Temporal median subtract -----------------------------------------
    med = jnp.median(clip, axis=0)                                             # (H, W)
    clip_median = clip - med[None]

    return dict(
        clip_raw=clip,
        clip_median=clip_median,
        temporal_median=med,           # (H, W) — the static context (Option A input)
        cells=cells,
        flagella=flag,
        pipette=pip,
        curves=curves,
        bg_level=bg_level,
        distortions=dict(faint=faint, defocus_sigma=defocus_sigma,
                          motion_k=motion_k, vig_strength=vig_strength,
                          noise_sigma=noise_sigma),
    )


def sample_batch(key: jax.Array, cfg: DiverseSimConfig, batch_size: int) -> dict:
    keys = jax.random.split(key, batch_size)
    return jax.vmap(lambda k: sample_clip(k, cfg))(keys)
