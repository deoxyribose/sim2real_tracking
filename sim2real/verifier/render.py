"""Rasterize candidate primitives on the canonical canvas.

Reuses the sim's flagellum + cell rendering functions, so a candidate has the
SAME appearance model as a sim-generated GT would. This is critical for the
inverse-rendering scoring to make sense.
"""
from __future__ import annotations

import numpy as np

from ..data.sim_flagella import _render_cell_frame, _render_flagellum_frame
from ..data import CANONICAL_H, CANONICAL_W


def render_flagellum_candidate(pts_incl_attach: np.ndarray, width_px: float, amp_signed: float,
                                H: int = CANONICAL_H, W: int = CANONICAL_W) -> np.ndarray:
    """One flagellum candidate → (H, W) signed contribution.

    pts_incl_attach: (K+1, 2) yx.
    amp_signed: +ve means darker than BG (negative pixel deviation), -ve means brighter.
    """
    # Note: sim uses `amp = -params["amp"]` so passing amp_signed here means "peak intensity
    # in the same canonical-residual sign convention". We render with amp = -amp_signed to
    # match the sim renderer's convention (darker → negative canonical residual).
    return _render_flagellum_frame(pts_incl_attach, width_px, -amp_signed, H, W)


def render_cell_candidate(center_yx: np.ndarray, radius_px: float, amp_signed: float,
                           H: int = CANONICAL_H, W: int = CANONICAL_W) -> np.ndarray:
    """One cell candidate → (H, W) signed contribution."""
    return _render_cell_frame(center_yx, float(radius_px), float(amp_signed), H, W)
