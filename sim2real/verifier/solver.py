"""CP-SAT integer program for discrete inverse rendering of flagellum + cell candidates.

Decision variables (per canonicalized clip):
  x_f[i] ∈ {0, 1}   — flagellum candidate i selected
  x_c[j] ∈ {0, 1}   — cell candidate j selected

Constraints:
  1. Non-overlap on FLAGELLUM candidates: for each pixel touched by >1 flagellum candidate,
     `sum of x_f[i] where c_i covers pixel` ≤ 1.  (Approximated via per-candidate pixel
     signature buckets, following the discrete_linking_opt at-most-one construction.)
  2. Attachment-on-cell: each selected flagellum's attachment must be within
     `attach_slack_px` of some selected cell's radius (soft — enforced via big-M).
  3. Cardinality caps: at most `max_cells` cells and `max_flagella` flagella per clip.

Objective (minimize):
  Σ_i cost_f[i] · x_f[i]  +  Σ_j cost_c[j] · x_c[j]  +  birth_prior · Σ x_f[i]

Where cost_* is the reconstruction delta computed by `score.reconstruction_delta_per_candidate`.
`birth_prior > 0` encodes a prior against over-explaining pixels with many candidates.

Adapted from /home/frans/discrete_linking_opt/3_solve_discrete_problem.py::build_model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from ortools.sat.python import cp_model


@dataclass
class VerifyConfig:
    max_cells: int = 2                    # per clip
    max_flagella: int = 4                 # per clip (2 cells × 2 flagella each)
    attach_slack_px: float = 4.0          # allowed distance from cell boundary to attachment
    overlap_thresh_sigma: float = 1.5     # pixel is "owned" by candidate if |render| > thresh
    birth_prior_per_flag: float = 0.5     # small positive cost per selected flagellum (prior)
    birth_prior_per_cell: float = 1.0     # bigger prior per cell (they're rare, we want few)
    require_cell_for_flag: bool = True    # if True, every flag must have a matching cell
    solver_time_limit_s: float = 30.0
    solver_workers: int = 8
    # Score integer scaling — CP-SAT needs integer coefficients
    score_scale: int = 100
    # Scoring mode: "recon" for L1 reconstruction delta, "matched" for negative inner product
    scoring: str = "matched"


def _build_pixel_signature_to_candidates(
    candidate_renders: np.ndarray, thresh: float
) -> dict[tuple[int, int], list[int]]:
    """For each pixel (y, x), the list of flagellum candidate indices that 'touch' it
    (|render| > thresh). Used to build non-overlap constraints."""
    N, H, W = candidate_renders.shape
    touched = np.abs(candidate_renders) > thresh                          # (N, H, W)
    # For efficiency: only iterate pixels touched by >= 2 candidates
    counts = touched.sum(axis=0)                                          # (H, W)
    contested = np.where(counts >= 2)
    out: dict[tuple[int, int], list[int]] = {}
    for y, x in zip(contested[0], contested[1]):
        cand_ids = np.where(touched[:, y, x])[0].tolist()
        out[(int(y), int(x))] = cand_ids
    return out


def _cell_attachment_compat_matrix(
    flag_attachments: np.ndarray,       # (Nf, 2)
    cell_centers: np.ndarray,           # (Nc, 2)
    cell_radii: np.ndarray,             # (Nc,)
    slack: float,
) -> np.ndarray:
    """Return (Nf, Nc) bool: True if flag i's attachment is within `slack` px of cell j's boundary."""
    # For each (i, j): distance to center; check if |dist - radius| <= slack
    d = np.linalg.norm(flag_attachments[:, None] - cell_centers[None], axis=-1)  # (Nf, Nc)
    return np.abs(d - cell_radii[None]) <= slack


def build_and_solve(
    flag_candidates: dict,
    cell_candidates: dict,
    target: np.ndarray,
    cfg: VerifyConfig = None,
    verbose: bool = False,
    energy_map: np.ndarray = None,
) -> dict:
    """
    flag_candidates: dict with keys
        - renders: (Nf, H, W) float32 signed contributions
        - attachments: (Nf, 2)
        - pts: (Nf, K+1, 2) (for output convenience)
        - width: (Nf,)  (for output)
        - amp: (Nf,)    (for output)
        - source_slot: (Nf,) int  (which model slot this came from — for output only)
    cell_candidates: dict with keys
        - renders: (Nc, H, W) float32
        - centers: (Nc, 2)
        - radii: (Nc,)
    target: (H, W) canonical residual to explain.
    cfg: VerifyConfig or None (defaults).

    Returns dict with:
        selected_flag_idx:  list[int]
        selected_cell_idx:  list[int]
        objective: int
        wall_time_s: float
        status: str
    """
    if cfg is None:
        cfg = VerifyConfig()

    Nf = flag_candidates["renders"].shape[0]
    Nc = cell_candidates["renders"].shape[0]

    # Per-candidate cost. Scoring modes: "matched", "recon", "energy".
    from .score import (
        matched_filter_cost, reconstruction_delta_per_candidate, energy_alignment_cost,
    )
    baseline = np.zeros_like(target)
    if cfg.scoring == "energy":
        if energy_map is None:
            raise ValueError("scoring='energy' requires energy_map to be provided.")
        flag_cost = energy_alignment_cost(flag_candidates["renders"], energy_map)
        cell_cost = energy_alignment_cost(cell_candidates["renders"], energy_map)
    elif cfg.scoring == "matched":
        flag_cost = matched_filter_cost(flag_candidates["renders"], target, baseline)
        cell_cost = matched_filter_cost(cell_candidates["renders"], target, baseline)
    else:
        flag_cost = reconstruction_delta_per_candidate(flag_candidates["renders"], target, baseline)
        cell_cost = reconstruction_delta_per_candidate(cell_candidates["renders"], target, baseline)

    # Add birth priors — penalizes over-selection (equivalent to a per-candidate constant)
    flag_cost_int = np.round((flag_cost + cfg.birth_prior_per_flag) * cfg.score_scale).astype(np.int64)
    cell_cost_int = np.round((cell_cost + cfg.birth_prior_per_cell) * cfg.score_scale).astype(np.int64)

    model = cp_model.CpModel()
    x_f = [model.NewBoolVar(f"xf_{i}") for i in range(Nf)]
    x_c = [model.NewBoolVar(f"xc_{j}") for j in range(Nc)]

    # Objective: minimize costs (adding a candidate LOWERS cost if it reduces reconstruction error)
    model.Minimize(
        sum(int(flag_cost_int[i]) * x_f[i] for i in range(Nf))
        + sum(int(cell_cost_int[j]) * x_c[j] for j in range(Nc))
    )

    # Cardinality caps
    if Nf > 0:
        model.Add(sum(x_f) <= cfg.max_flagella)
    if Nc > 0:
        model.Add(sum(x_c) <= cfg.max_cells)

    # Non-overlap on flagella (pixel-signature at-most-one)
    if Nf > 1:
        contested = _build_pixel_signature_to_candidates(
            flag_candidates["renders"], thresh=cfg.overlap_thresh_sigma)
        # Deduplicate constraint groups (same set of candidates → one constraint)
        seen: set[tuple[int, ...]] = set()
        for pix, cand_ids in contested.items():
            key = tuple(sorted(cand_ids))
            if key in seen:
                continue
            seen.add(key)
            if len(cand_ids) >= 2:
                model.AddAtMostOne([x_f[i] for i in cand_ids])

    # Non-overlap on cells (rarely needed since cells are big and few, but include for correctness)
    if Nc > 1:
        contested_c = _build_pixel_signature_to_candidates(
            cell_candidates["renders"], thresh=cfg.overlap_thresh_sigma * 2)  # cells are bigger
        seen_c: set[tuple[int, ...]] = set()
        for _, cand_ids in contested_c.items():
            key = tuple(sorted(cand_ids))
            if key in seen_c:
                continue
            seen_c.add(key)
            if len(cand_ids) >= 2:
                model.AddAtMostOne([x_c[j] for j in cand_ids])

    # Attachment-on-cell: for each flag i, x_f[i] = 1 implies at least one x_c[j] = 1 s.t. compat[i, j]
    if cfg.require_cell_for_flag and Nc > 0 and Nf > 0:
        compat = _cell_attachment_compat_matrix(
            flag_candidates["attachments"], cell_candidates["centers"],
            cell_candidates["radii"], cfg.attach_slack_px,
        )
        for i in range(Nf):
            compat_cells = np.where(compat[i])[0].tolist()
            if not compat_cells:
                # No cell can host this flagellum; forbid it entirely.
                model.Add(x_f[i] == 0)
            else:
                # x_f[i] ≤ sum(x_c[j] for j in compat_cells)  → attachment implies at least one host
                model.Add(x_f[i] <= sum(x_c[j] for j in compat_cells))

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = cfg.solver_time_limit_s
    solver.parameters.num_workers = cfg.solver_workers
    if verbose:
        solver.parameters.log_search_progress = True

    status = solver.Solve(model)
    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }.get(status, str(status))

    selected_flag = [i for i in range(Nf) if solver.Value(x_f[i]) == 1] if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else []
    selected_cell = [j for j in range(Nc) if solver.Value(x_c[j]) == 1] if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else []
    obj = solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None

    return dict(
        selected_flag_idx=selected_flag,
        selected_cell_idx=selected_cell,
        objective=float(obj) if obj is not None else None,
        wall_time_s=float(solver.WallTime()),
        status=status_name,
        n_flag_candidates=Nf,
        n_cell_candidates=Nc,
    )
