"""Top-level DETR-slot model. Wires encoder → slot attention → typed heads."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from .encoder import Encoder
from .heads import CellHead, ClassHead, FlagellumHead
from .slot_attention import SlotAttention


@dataclass(frozen=True)
class DETRSlotConfig:
    n_slots: int = 8
    n_iters: int = 3
    d_slot: int = 128
    d_ff_slot: int = 256
    d_ff_head: int = 256
    encoder_d_model: int = 128
    encoder_channels: tuple[int, ...] = (32, 64, 128)
    encoder_strides: tuple[int, ...] = (2, 2, 2)


class DETRSlotModel(nn.Module):
    cfg: DETRSlotConfig = field(default_factory=DETRSlotConfig)

    @nn.compact
    def __call__(self, clip: jnp.ndarray, energy: jnp.ndarray) -> dict:
        """clip: (B, T, H, W). energy: (B, H, W). Returns dict of head outputs.

        Keys:
          class_logits:  (B, n_slots, N_CLASSES)
          pts_mean:      (B, n_slots, K+1, 2)
          pts_log_sigma: (B, n_slots, K+1, 2)
          width_mean, width_log_sigma:  (B, n_slots)
          amp_mean,   amp_log_sigma:    (B, n_slots)
          polarity_logit:               (B, n_slots)
        """
        features, _grid = Encoder(
            d_model=self.cfg.encoder_d_model,
            conv_channels=self.cfg.encoder_channels,
            conv_strides=self.cfg.encoder_strides,
            name="encoder",
        )(clip, energy)

        slots = SlotAttention(
            n_slots=self.cfg.n_slots, n_iters=self.cfg.n_iters,
            d_slot=self.cfg.d_slot, d_ff=self.cfg.d_ff_slot,
            name="slot_attn",
        )(features)

        class_logits = ClassHead(d_ff=self.cfg.d_ff_head, name="class_head")(slots)
        flag_out = FlagellumHead(d_ff=self.cfg.d_ff_head, name="flagellum_head")(slots)
        cell_out = CellHead(d_ff=self.cfg.d_ff_head // 2, name="cell_head")(slots)
        return dict(class_logits=class_logits, **flag_out, **cell_out)
