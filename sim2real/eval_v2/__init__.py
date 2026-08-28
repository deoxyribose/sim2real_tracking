"""Evaluation for the DETR-slot v2 model."""
from .coverage import (
    load_real_annotations,
    sample_coverage_recall,
    canonicalize_real_frame,
)

__all__ = [
    "load_real_annotations",
    "sample_coverage_recall",
    "canonicalize_real_frame",
]
