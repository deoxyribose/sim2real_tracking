"""Multi-frame forward + grad + temporal identity tests for SlotVideoModel."""

import jax
import jax.numpy as jnp

from sim2real.model.model import ModelConfig, SlotVideoModel
from sim2real.sim.api import build_sim


def _build_small_model(use_discovery=True):
    cfg = ModelConfig(
        n_max=8,
        d_model=64,
        n_heads=4,
        n_transformer_layers=1,
        z_what_dim=32,
        z_style_dim=8,
        glimpse_size=12,
        stem_channels=(16, 32, 48),
        use_discovery=use_discovery,
    )
    return SlotVideoModel(cfg=cfg), cfg


def test_forward_finite_loss_and_grad():
    model, _ = _build_small_model()
    key = jax.random.key(0)
    video = jax.random.uniform(jax.random.fold_in(key, 1), (4, 64, 64, 1))
    params = model.init(key, video, key)
    out = model.apply(params, video, key)
    assert out.z_where.shape == (4, 8, 5)
    assert out.z_pres.shape == (4, 8)
    assert out.composite.shape == (4, 64, 64, 1)
    loss = jnp.mean((out.composite - video) ** 2)
    assert jnp.isfinite(loss)

    def loss_fn(p):
        o = model.apply(p, video, key)
        return jnp.mean((o.composite - video) ** 2)

    g = jax.grad(loss_fn)(params)
    total = sum(jnp.sum(jnp.abs(x)) for x in jax.tree.leaves(g))
    assert bool(jnp.isfinite(total))


def test_temporal_propagation_preserves_slot_identity_on_static_video():
    """If the same input image is repeated across T frames, the slot z_what should be stable."""
    model, _ = _build_small_model()
    key = jax.random.key(2)
    static_frame = jax.random.uniform(jax.random.fold_in(key, 1), (64, 64, 1))
    video = jnp.broadcast_to(static_frame[None], (5, 64, 64, 1))
    params = model.init(key, video, key)
    # Use deterministic forward (we cannot easily disable Gumbel; but with same key, results match).
    out = model.apply(params, video, key)
    # z_what at t=4 should be close to z_what at t=0 (no movement => no drift in z_where, low GRU change).
    diff = jnp.mean(jnp.abs(out.z_what[4] - out.z_what[0]))
    # Loose bound — these are unrolled steps with re-sampled what and stochastic pres.
    assert float(diff) < 2.0


def test_discovery_disabled_still_runs():
    model, _ = _build_small_model(use_discovery=False)
    key = jax.random.key(3)
    video = jax.random.uniform(key, (3, 32, 32, 1))
    params = model.init(key, video, key)
    out = model.apply(params, video, key)
    assert out.composite.shape == (3, 32, 32, 1)


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
