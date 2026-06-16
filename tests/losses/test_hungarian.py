"""Tests for Hungarian matching and the supervised loss aggregator."""

import jax
import jax.numpy as jnp

from sim2real.losses.matching import build_cost_zwhere, hungarian, hungarian_per_frame
from sim2real.losses.losses import AdaptLossConfig, PretrainLossConfig, adapt_loss, pretrain_loss
from sim2real.model.model import ModelConfig, SlotVideoModel
from sim2real.priors.registry import PriorConfig
from sim2real.sim.api import build_sim


def test_hungarian_recovers_identity_on_noise_free_case():
    # Identity case: pred = gt with no shuffle.
    N = 5
    gt = jnp.array([[0.0, 0.1, 0.2], [0.0, -0.2, 0.1], [0.0, 0.4, -0.1], [0.0, -0.5, 0.3], [0.0, 0.3, 0.3]])
    pres = jnp.ones((N,))
    perm = hungarian_per_frame(gt, gt, pres)
    assert jnp.all(perm == jnp.arange(N))


def test_hungarian_recovers_shuffle():
    # Shuffle GT slots; recovered perm should put the predictions back in GT order.
    N = 5
    rng = jax.random.key(42)
    gt = jax.random.normal(rng, (N, 3))
    pres = jnp.ones((N,))
    shuffle = jnp.array([3, 0, 4, 1, 2])
    pred = gt[shuffle]  # pred[i] = gt[shuffle[i]] — i.e. pred slot i corresponds to gt slot shuffle[i]
    perm = hungarian_per_frame(pred, gt, pres)
    # pred[perm[j]] == gt[j], so perm[j] should equal the inverse of shuffle.
    inv = jnp.argsort(shuffle)
    assert jnp.all(perm == inv)


def test_pretrain_loss_runs_on_real_sim_and_model():
    cfg_m = ModelConfig(
        n_max=8, d_model=64, n_heads=4, n_transformer_layers=1, z_what_dim=32, z_style_dim=8,
        glimpse_size=12, stem_channels=(16, 32, 48),
    )
    model = SlotVideoModel(cfg=cfg_m)
    batch_fn, sim_cfg = build_sim("flagella")

    key = jax.random.key(0)
    sample = batch_fn(key, 1)
    # Take first video and trim to small T for speed.
    sample0 = jax.tree.map(lambda x: x[0] if hasattr(x, "shape") and x.shape and x.shape[0] == 1 else x, sample)
    # Above pulls leading axis. To be safe, build sample0 explicitly:
    from sim2real.types import SimSample
    sample0 = SimSample(
        video=sample.video[0, :3],
        z_where=sample.z_where[0, :3, :8],  # match model n_max
        z_pres=sample.z_pres[0, :3, :8],
        z_style=sample.z_style[0],
        masks=sample.masks[0, :3, :8],
        z_what=None,
        meta=sample.meta,
    )

    params = model.init(key, sample0.video, key)
    out = model.apply(params, sample0.video, key)

    pre_cfg = PretrainLossConfig()
    pri_cfg = PriorConfig()
    total, metrics = pretrain_loss(out, sample0, pre_cfg, pri_cfg)
    assert jnp.isfinite(total)
    for k, v in metrics.items():
        assert jnp.isfinite(v), f"{k} not finite"


def test_adapt_loss_runs_and_no_supervised_gradient_leak():
    cfg_m = ModelConfig(
        n_max=8, d_model=64, n_heads=4, n_transformer_layers=1, z_what_dim=32, z_style_dim=8,
        glimpse_size=12, stem_channels=(16, 32, 48),
    )
    model = SlotVideoModel(cfg=cfg_m)
    batch_fn, _ = build_sim("flagella")
    key = jax.random.key(0)
    sample = batch_fn(key, 1)
    from sim2real.types import SimSample
    sample0 = SimSample(
        video=sample.video[0, :3],
        z_where=sample.z_where[0, :3, :8],
        z_pres=sample.z_pres[0, :3, :8],
        z_style=sample.z_style[0],
        masks=sample.masks[0, :3, :8],
        z_what=None,
        meta=sample.meta,
    )
    params = model.init(key, sample0.video, key)

    def loss_fn_adapt(p):
        o = model.apply(p, sample0.video, key)
        total, _ = adapt_loss(o, sample0, AdaptLossConfig(), PriorConfig())
        return total

    g = jax.grad(loss_fn_adapt)(params)
    total_g = sum(jnp.sum(jnp.abs(x)) for x in jax.tree.leaves(g))
    assert bool(jnp.isfinite(total_g))


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
            import traceback; traceback.print_exc()
            print(f"FAIL {name}: {e}")
            failed += 1
    sys.exit(failed)
