"""Aggregated prior config + KL accumulator.

Every training run uses the same set of priors; their weights and hyperparameters live here so
the loss code remains thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sim2real.priors.pres import PresPriorConfig
from sim2real.priors.what import WhatPriorConfig
from sim2real.priors.where import WherePriorConfig


@dataclass(frozen=True)
class PriorConfig:
    where: WherePriorConfig = field(default_factory=WherePriorConfig)
    pres: PresPriorConfig = field(default_factory=PresPriorConfig)
    what: WhatPriorConfig = field(default_factory=WhatPriorConfig)
    weight_where: float = 1.0
    weight_pres: float = 0.1
    weight_what_temporal: float = 0.1
    weight_what_cross: float = 0.1
    weight_style: float = 1.0
