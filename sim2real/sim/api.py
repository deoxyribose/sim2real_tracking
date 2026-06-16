"""Simulator registry — `build_sim('flagella') -> (sample_fn, default_cfg)`."""

from __future__ import annotations

from typing import Callable

import jax

from sim2real.sim.configs import FlagellaConfig, ManyCellsConfig, MultiScaleConfig, WormsConfig
from sim2real.sim.flagella import sample as flagella_sample, sample_batch as flagella_batch
from sim2real.sim.many_cells import sample as many_cells_sample, sample_batch as many_cells_batch
from sim2real.sim.multiscale import sample as multiscale_sample, sample_batch as multiscale_batch
from sim2real.sim.worms import sample as worms_sample, sample_batch as worms_batch
from sim2real.types import SimSample

_REGISTRY: dict[str, tuple[Callable, Callable, type]] = {
    "flagella": (flagella_sample, flagella_batch, FlagellaConfig),
    "many_cells": (many_cells_sample, many_cells_batch, ManyCellsConfig),
    "multiscale": (multiscale_sample, multiscale_batch, MultiScaleConfig),
    "worms": (worms_sample, worms_batch, WormsConfig),
}


def list_sims() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_default_config(kind: str):
    if kind not in _REGISTRY:
        raise KeyError(f"unknown sim {kind}; choose one of {list_sims()}")
    return _REGISTRY[kind][2]()


def get_sample_fn(kind: str) -> Callable[[jax.Array, object], SimSample]:
    return _REGISTRY[kind][0]


def get_batch_fn(kind: str) -> Callable[[jax.Array, object, int], SimSample]:
    return _REGISTRY[kind][1]


def build_sim(kind: str, cfg=None) -> tuple[Callable[[jax.Array, int], SimSample], object]:
    """Return a closure `batch_fn(key, batch_size) -> SimSample` with the given config baked in."""
    sample_batch_fn = get_batch_fn(kind)
    if cfg is None:
        cfg = get_default_config(kind)

    def fn(key: jax.Array, batch_size: int) -> SimSample:
        return sample_batch_fn(key, cfg, batch_size)

    return fn, cfg
