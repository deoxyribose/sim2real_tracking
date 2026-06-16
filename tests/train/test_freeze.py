"""Tests for path-glob freeze masks."""

import jax
import jax.numpy as jnp
import optax

from sim2real.model.model import ModelConfig, SlotVideoModel
from sim2real.train.freeze import build_freeze_mask, make_optimizer_with_freeze


def test_build_mask_matches_encoder():
    cfg = ModelConfig(
        n_max=4, d_model=32, n_heads=2, n_transformer_layers=1, z_what_dim=16, z_style_dim=4,
        glimpse_size=8, stem_channels=(8, 16, 24),
    )
    model = SlotVideoModel(cfg=cfg)
    key = jax.random.key(0)
    video = jnp.zeros((2, 32, 32, 1))
    params = model.init(key, video, key)
    mask = build_freeze_mask(params, ["encoder.*"])
    flat = jax.tree_util.tree_leaves_with_path(mask)
    n_frozen = sum(int(bool(v)) for _, v in flat)
    n_total = len(flat)
    # Some leaves should be frozen, some should not.
    assert 0 < n_frozen < n_total


def test_frozen_params_dont_update():
    cfg = ModelConfig(
        n_max=4, d_model=32, n_heads=2, n_transformer_layers=1, z_what_dim=16, z_style_dim=4,
        glimpse_size=8, stem_channels=(8, 16, 24),
    )
    model = SlotVideoModel(cfg=cfg)
    key = jax.random.key(0)
    video = jnp.zeros((2, 32, 32, 1))
    params = model.init(key, video, key)
    base_opt = optax.adam(1e-3)
    opt, mask = make_optimizer_with_freeze(base_opt, params, ["encoder.*"])
    opt_state = opt.init(params)

    def loss(p):
        out = model.apply(p, video, key)
        return jnp.mean(out.composite ** 2)

    g = jax.grad(loss)(params)
    updates, _ = opt.update(g, opt_state, params)

    # Check that frozen leaves have zero updates and non-frozen have non-zero updates.
    flat_updates = jax.tree_util.tree_leaves_with_path(updates)
    flat_mask = jax.tree_util.tree_leaves_with_path(mask)

    for (path_u, upd), (path_m, frozen) in zip(flat_updates, flat_mask):
        assert path_u == path_m
        if bool(frozen):
            assert float(jnp.sum(jnp.abs(upd))) == 0.0, f"frozen leaf {path_u} got nonzero update"


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
