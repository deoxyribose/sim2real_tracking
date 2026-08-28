"""DETR-with-slot-attention model (v2) — 2026-08-28 refactor.

Slot-attention encoder + typed distributional heads per slot. No decoder, no
reconstruction, no temporal identity in outputs. See [[refactor-2026-08-28]] for
context.
"""
from .detr_slot import DETRSlotModel, DETRSlotConfig
from .heads import (
    FlagellumHead, ClassHead, CellHead,
    class_and_flagellum_from_slot,
    sample_flagellum_from_head, sample_cell_from_head,
)
from .slot_attention import SlotAttention
from .loss import (
    MAX_GT_FLAG, MAX_GT_CELL,
    pack_gt_batch,
    hungarian_match_flagella, hungarian_match_cells,
    run_matches, compute_loss,
)

__all__ = [
    "DETRSlotModel", "DETRSlotConfig",
    "FlagellumHead", "ClassHead", "CellHead",
    "class_and_flagellum_from_slot",
    "sample_flagellum_from_head", "sample_cell_from_head",
    "SlotAttention",
    "MAX_GT_FLAG", "MAX_GT_CELL",
    "pack_gt_batch", "hungarian_match_flagella", "hungarian_match_cells",
    "run_matches", "compute_loss",
]
