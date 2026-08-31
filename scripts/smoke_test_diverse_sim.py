"""Smoke test: draw one clip from the diverse sim and dump its shape/stats.
No JAX tracing subtleties — just call it and print. If this survives, next
step is a viz grid."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip


def main():
    cfg = DiverseSimConfig(T=16, H=128, W=128, n_max_cells=3, n_max_flagella=6,
                           n_arc_pts=20)
    key = jax.random.key(0)
    print("compiling + running sample_clip...")
    out = jax.block_until_ready(sample_clip(key, cfg))
    print("done")
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                if hasattr(v2, "shape"):
                    print(f"    {k2}: {v2.shape} {v2.dtype}")
                else:
                    print(f"    {k2}: {v2}")
        elif hasattr(v, "shape"):
            print(f"  {k}: {v.shape} {v.dtype}  "
                  f"[{float(v.min()):.3f}, {float(v.max()):.3f}]  "
                  f"mean={float(v.mean()):.3f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
