"""Data preprocessing: canonicalization pipeline shared by real videos and sim composites."""
from .canonicalize import (
    CanonicalConfig,
    CANONICAL_H,
    CANONICAL_W,
    CANONICAL_TARGET_WIDTH_PX,
    canonicalize_clip,
    running_median_bg,
    band_pass_flagellum,
    resample_to_width,
    sigma_scale,
    temporal_energy_map,
)
from .types import (
    CLASS_EMPTY, CLASS_PIPETTE, CLASS_CELL, CLASS_FLAGELLUM, N_CLASSES,
    FLAGELLUM_K,
    CellLatent, FlagellumLatent, SceneLatents, SimSampleV2,
)
from .sim_flagella import FlagellumSimConfig, sample_scene

__all__ = [
    "CanonicalConfig",
    "CANONICAL_H", "CANONICAL_W",
    "CANONICAL_TARGET_WIDTH_PX",
    "canonicalize_clip",
    "running_median_bg", "band_pass_flagellum", "resample_to_width",
    "sigma_scale", "temporal_energy_map",
    "CLASS_EMPTY", "CLASS_PIPETTE", "CLASS_CELL", "CLASS_FLAGELLUM", "N_CLASSES",
    "FLAGELLUM_K",
    "CellLatent", "FlagellumLatent", "SceneLatents", "SimSampleV2",
    "FlagellumSimConfig", "sample_scene",
]
