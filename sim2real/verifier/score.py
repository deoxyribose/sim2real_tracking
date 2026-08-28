"""Discrete inverse-rendering scoring.

For each candidate rendering c_i, compute the change in per-pixel reconstruction
error if we ADD c_i to the current baseline:

    L(c_i, pixel) = |baseline + c_i(pixel) - target(pixel)| - |baseline - target(pixel)|

Negative L means the candidate REDUCES the reconstruction error at that pixel.
The per-candidate "delta cost" is the sum over all pixels of L(c_i, pixel).

Baseline is currently 0 (canonical residual is already zero-mean around a σ-scaled
noise floor); the target is the canonical clip's midpoint frame.

Adapted from `discrete_linking_opt/2_make_discrete_opt_problem.py::_frame_deltas_jax`
but scalar-target (grayscale residual) instead of RGB.
"""
from __future__ import annotations

import numpy as np


def reconstruction_delta_per_candidate(
    candidate_renders: np.ndarray,     # (N, H, W) signed contributions
    target: np.ndarray,                # (H, W) canonical residual to explain
    baseline: np.ndarray = None,       # (H, W) starting canvas (usually zeros)
) -> np.ndarray:
    """Per-candidate ΔL = L(baseline + c_i) - L(baseline).

    ΔL < 0 ↔ candidate improves reconstruction ↔ we want to select it.
    Returns (N,) float32 — the total per-candidate cost (negative is good).
    """
    N = candidate_renders.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=np.float32)
    H, W = candidate_renders.shape[1], candidate_renders.shape[2]
    if baseline is None:
        baseline = np.zeros((H, W), dtype=np.float32)
    base_err_sum = float(np.abs(baseline - target).sum())
    with np.errstate(over="ignore"):
        added_err = np.abs(baseline[None] + candidate_renders - target[None])  # (N, H, W)
    per_cand_err = added_err.reshape(N, -1).sum(axis=1)                        # (N,)
    return (per_cand_err - base_err_sum).astype(np.float32)


def per_pixel_delta_by_candidate(
    candidate_renders: np.ndarray,
    target: np.ndarray,
    baseline: np.ndarray = None,
) -> np.ndarray:
    """Return per-pixel ΔL for each candidate — (N, H, W). Useful for pixel-level constraints
    (e.g. non-overlap based on which candidates 'own' which pixel)."""
    N, H, W = candidate_renders.shape
    if baseline is None:
        baseline = np.zeros((H, W), dtype=np.float32)
    return (np.abs(baseline[None] + candidate_renders - target[None]) - np.abs(baseline - target)[None]).astype(np.float32)


def which_pixels_touched(candidate_render: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """Boolean mask of pixels where |c_i| > thresh — used for non-overlap constraints."""
    return np.abs(candidate_render) > thresh


def energy_alignment_cost(
    candidate_renders: np.ndarray,   # (N, H, W)
    energy: np.ndarray,               # (H, W) non-negative
    render_thresh: float = 1.0,
) -> np.ndarray:
    """Cost = -mean(energy at pixels touched by candidate).

    Rewards candidates that trace through high-temporal-energy pixels (where the
    flagellum actually beats). Works better than L1/matched-filter on real data because
    energy is a cleaner "flagellum-here" signal than raw residual intensity, which is
    contaminated by cell body, static structure, and noise.
    """
    N = candidate_renders.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=np.float32)
    masks = np.abs(candidate_renders) > render_thresh  # (N, H, W)
    # Per-candidate: mean energy at touched pixels (or 0 if no pixels)
    counts = masks.reshape(N, -1).sum(axis=1) + 1e-6
    total = (masks * energy[None]).reshape(N, -1).sum(axis=1)
    return (-total / counts).astype(np.float32)


def matched_filter_cost(
    candidate_renders: np.ndarray,   # (N, H, W)
    target: np.ndarray,               # (H, W)
    baseline: np.ndarray = None,
    normalize: bool = True,
) -> np.ndarray:
    """Cost = -(target · candidate_render) / sqrt(||render||²) — normalized matched filter.

    Without normalization, larger candidates dominate simply because they have more
    pixels contributing. Normalizing by ||render|| turns this into an angle-based score
    (higher when target and render are aligned in shape, not just size).
    """
    N = candidate_renders.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=np.float32)
    if baseline is not None:
        target = target - baseline
    flat_c = candidate_renders.reshape(N, -1)
    inner = (flat_c * target.reshape(-1)[None]).sum(axis=1)
    if normalize:
        norm = np.sqrt((flat_c ** 2).sum(axis=1) + 1e-6)
        return (-inner / norm).astype(np.float32)
    return (-inner).astype(np.float32)
