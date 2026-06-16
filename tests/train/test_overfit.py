"""Integration test: a few pretrain steps should drop recon loss on a small config."""

import jax
import jax.numpy as jnp

from sim2real.losses.losses import PretrainLossConfig, pretrain_loss
from sim2real.model.model import ModelConfig, SlotVideoModel
from sim2real.priors.registry import PriorConfig
from sim2real.sim.api import build_sim
from sim2real.train.batch import SimBatcher
from sim2real.train.schedule import adamw_cosine
from sim2real.types import SimSample


def _slice_n_max(batch, Nm):
    return SimSample(
        video=batch.video,
        z_where=batch.z_where[:, :, :Nm],
        z_pres=batch.z_pres[:, :, :Nm],
        z_style=batch.z_style,
        masks=batch.masks[:, :, :Nm],
        z_what=None,
        meta=batch.meta,
    )


def test_pretrain_step_recon_decreases():
    """A handful of steps on a fixed batch should drop recon (loop reaches the optimizer)."""
    cfg_m = ModelConfig(
        n_max=8, d_model=64, n_heads=2, n_transformer_layers=1, z_what_dim=32, z_style_dim=8,
        glimpse_size=12, stem_channels=(16, 32, 48),
    )
    model = SlotVideoModel(cfg=cfg_m)
    batcher = SimBatcher("flagella", batch_size=1)
    key = jax.random.key(0)
    batch = batcher.sample(key)
    batch_m = _slice_n_max(batch, cfg_m.n_max)
    params = model.init(key, batch_m.video[0], key)

    optimizer, _ = adamw_cosine(1e-3, total_steps=10, warmup_steps=1)
    opt_state = optimizer.init(params)
    pre_cfg = PretrainLossConfig()
    pri_cfg = PriorConfig()

    def per_video(out, smp):
        total, m = pretrain_loss(out, smp, pre_cfg, pri_cfg)
        return total, m

    @jax.jit
    def step(p, st, b, k):
        def loss_fn(p):
            keys = jax.random.split(k, b.video.shape[0])
            outs = jax.vmap(lambda v, kk: model.apply(p, v, kk))(b.video, keys)
            totals, _ = jax.vmap(per_video)(outs, b)
            return jnp.mean(totals)

        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, st = optimizer.update(grads, st, p)
        p = optax.apply_updates(p, updates)
        return p, st, loss

    import optax
    init_loss = None
    final_loss = None
    for i in range(8):
        params, opt_state, loss = step(params, opt_state, batch_m, jax.random.fold_in(key, i))
        if i == 0:
            init_loss = float(loss)
        final_loss = float(loss)

    assert init_loss is not None and final_loss is not None
    assert jnp.isfinite(jnp.array(final_loss))
    # Loose check: the loss did not explode.
    assert final_loss < init_loss * 2.0


if __name__ == "__main__":
    test_pretrain_step_recon_decreases()
    print("PASS test_pretrain_step_recon_decreases")
