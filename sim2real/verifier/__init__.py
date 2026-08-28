"""Discrete verifier: renders model-sampled candidate curves, scores each against
the canonical clip via reconstruction delta, then selects a non-overlapping subset
that best explains the pixels via CP-SAT integer programming.

Adapted from /home/frans/discrete_linking_opt (Meyer et al.-style discrete inverse
rendering for cells / worms) but specialized for our:
  - Flagellum tube rendering (short curved 1D structures)
  - Cell body renderer (dark disks)
  - Attachment-on-cell constraint (built-in via IP)
  - Canonical residual clip pixel space (rather than RGB)
"""
from .render import render_flagellum_candidate, render_cell_candidate
from .score import reconstruction_delta_per_candidate
from .solver import build_and_solve, VerifyConfig

__all__ = [
    "render_flagellum_candidate",
    "render_cell_candidate",
    "reconstruction_delta_per_candidate",
    "build_and_solve",
    "VerifyConfig",
]
