"""Shape/range contract tests for every simulator."""

import jax
import jax.numpy as jnp

from sim2real.sim.api import build_sim, list_sims


def _check_sample(kind: str, batch_size: int = 2):
    batch_fn, cfg = build_sim(kind)
    key = jax.random.key(0)
    s = jax.jit(lambda k: batch_fn(k, batch_size))(key)
    cm = cfg.common
    H = W = cm.res
    T = cm.T
    N = cm.n_max

    assert s.video.shape == (batch_size, T, H, W, 1), f"{kind} video shape {s.video.shape}"
    assert s.z_where.shape == (batch_size, T, N, 5)
    assert s.z_pres.shape == (batch_size, T, N)
    assert s.masks.shape == (batch_size, T, N, H, W)
    assert s.z_style.shape == (batch_size, cm.style_dim)

    # Range checks
    assert 0.0 <= float(s.video.min()) and float(s.video.max()) <= 1.0001
    assert 0.0 <= float(s.masks.min()) and float(s.masks.max()) <= 1.0001
    # z_pres is binary
    assert jnp.allclose(s.z_pres, s.z_pres.astype(jnp.int32).astype(jnp.float32))
    # z_where finite
    assert bool(jnp.all(jnp.isfinite(s.z_where)))


def test_flagella():
    _check_sample("flagella")


def test_many_cells():
    _check_sample("many_cells")


def test_multiscale():
    _check_sample("multiscale")


def test_worms():
    _check_sample("worms")


def test_all_sims_listed():
    assert set(list_sims()) == {"flagella", "many_cells", "multiscale", "worms"}


if __name__ == "__main__":
    import inspect, sys
    failed = 0
    for name, fn in inspect.getmembers(sys.modules[__name__], inspect.isfunction):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {e}")
            failed += 1
    sys.exit(failed)
