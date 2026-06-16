"""Batched simulator sampler.

For the smoke-test scale (`batch_size ≤ 8`, T ≤ 30) we sample synchronously inside the train step.
A prefetch / async-host pipeline can be added later (`flax.jax_utils.prefetch_to_device`).
"""

from __future__ import annotations

import jax

from sim2real.sim.api import build_sim
from sim2real.types import SimSample


class SimBatcher:
    def __init__(self, sim_kind: str, batch_size: int, cfg=None):
        self.batch_fn, self.cfg = build_sim(sim_kind, cfg)
        self.batch_size = batch_size
        self.kind = sim_kind

    def sample(self, key) -> SimSample:
        return self.batch_fn(key, self.batch_size)

    def jit_sample(self):
        """Return a `jit`-compiled closure that produces a batch given a key."""
        bs = self.batch_size
        fn = self.batch_fn

        @jax.jit
        def go(k):
            return fn(k, bs)

        return go
