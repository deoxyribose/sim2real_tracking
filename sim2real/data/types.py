"""Typed latent formats emitted by the new sim + predicted by the new model.

Per slot, one of:
  - Empty (∅ class) — no other params
  - Flagellum — attachment point + K arc-length control points + width + polarity
  - Cell     — center + radius (TODO — not implemented in v0)
  - Pipette  — line/capsule params (TODO — not implemented in v0)

For MVP we only support Flagellum. Cell and pipette are stubbed out (class=cell/pipette
plus a mean-only param head) and will be filled in later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Class ids. Slot 0 is background/empty. Keep integers small — one-hot in logits.
CLASS_EMPTY = 0
CLASS_PIPETTE = 1
CLASS_CELL = 2
CLASS_FLAGELLUM = 3
N_CLASSES = 4

# Number of arc-length control points per flagellum. Attachment counts as point 0, so
# a flagellum's curve is represented by K+1 (y, x) samples in total. Chosen so that the
# median-length flagellum (50 px) has one control point per ~7 px of arc length.
FLAGELLUM_K = 7   # 7 control points + attachment = 8 (y, x) samples per flagellum


@dataclass
class FlagellumLatent:
    """Ground-truth latent for one flagellum in canonical space.

    Coordinates are in canonical PIXEL units on the canonical (H_c, W_c) canvas —
    the same space the model sees at its input.
    """
    attachment: np.ndarray            # (2,) float32 = (y, x) canonical px
    control_points: np.ndarray        # (K, 2) float32 = (y, x), arc-length-ordered from attachment outward
    width_px: float                   # tube thickness (canonical px)
    polarity: int                     # +1 = darker than BG, -1 = brighter
    amplitude_sigma: float            # peak signed intensity in canonical σ-units

    def all_points(self) -> np.ndarray:
        """Return (K+1, 2) with attachment prepended."""
        return np.concatenate([self.attachment[None], self.control_points], axis=0)

    def arc_length_px(self) -> float:
        pts = self.all_points()
        diffs = np.diff(pts, axis=0)
        return float(np.sqrt((diffs ** 2).sum(-1)).sum())


@dataclass
class SceneLatents:
    """All ground-truth latents for one clip. Anonymous per-clip candidate set.

    Slots are un-ordered — no identity across clips. Any slot can be any class.
    For MVP, `flagella` is the only populated list; `cells` and `pipettes` are
    empty placeholders.
    """
    flagella: list[FlagellumLatent] = field(default_factory=list)
    # Populated in later phases:
    cells: list = field(default_factory=list)
    pipettes: list = field(default_factory=list)

    @property
    def n_objects(self) -> int:
        return len(self.flagella) + len(self.cells) + len(self.pipettes)


@dataclass
class SimSampleV2:
    """One canonicalized sim clip + its GT latents. Same schema real videos will produce
    at validation time (with .latents=None)."""
    clip: np.ndarray                  # (T, H_c, W_c) float32, canonical residual (σ-scaled, clipped)
    energy: np.ndarray                # (H_c, W_c) float32, temporal-energy map
    latents: SceneLatents             # ground-truth per-object latents
    sigma_canonical: float            # noise σ of this clip's canonical space
