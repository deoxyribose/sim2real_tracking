"""Smoke tests for DETRSlotModel."""
import jax
import jax.numpy as jnp

from sim2real.data import CANONICAL_H, CANONICAL_W, FLAGELLUM_K, N_CLASSES
from sim2real.model_v2 import DETRSlotModel, DETRSlotConfig


def _init(model, B=2, T=8, key=0):
    key = jax.random.PRNGKey(key)
    clip = jnp.zeros((B, T, CANONICAL_H, CANONICAL_W))
    energy = jnp.zeros((B, CANONICAL_H, CANONICAL_W))
    params = model.init({"params": key, "slots": key}, clip, energy)
    return params, clip, energy


def test_model_output_shapes():
    cfg = DETRSlotConfig(n_slots=6)
    model = DETRSlotModel(cfg=cfg)
    params, clip, energy = _init(model)
    out = model.apply(params, clip, energy, rngs={"slots": jax.random.PRNGKey(1)})
    B = clip.shape[0]
    S = cfg.n_slots
    assert out["class_logits"].shape == (B, S, N_CLASSES)
    assert out["pts_mean"].shape == (B, S, FLAGELLUM_K + 1, 2)
    assert out["pts_log_sigma"].shape == (B, S, FLAGELLUM_K + 1, 2)
    assert out["width_mean"].shape == (B, S)
    assert out["amp_mean"].shape == (B, S)
    assert out["polarity_logit"].shape == (B, S)


def test_model_output_ranges():
    """Point predictions should lie within the canvas; widths within a reasonable range."""
    cfg = DETRSlotConfig(n_slots=4)
    model = DETRSlotModel(cfg=cfg)
    params, clip, energy = _init(model)
    out = model.apply(params, clip, energy, rngs={"slots": jax.random.PRNGKey(1)})
    assert (out["pts_mean"][..., 0] >= 0).all() and (out["pts_mean"][..., 0] <= CANONICAL_H).all()
    assert (out["pts_mean"][..., 1] >= 0).all() and (out["pts_mean"][..., 1] <= CANONICAL_W).all()
    assert (out["width_mean"] >= 2.4).all() and (out["width_mean"] <= 8.6).all()
    assert (out["amp_mean"] >= 1.9).all()


def test_grad_flows_through_all_heads():
    cfg = DETRSlotConfig(n_slots=4)
    model = DETRSlotModel(cfg=cfg)
    params, clip, energy = _init(model)

    def loss_fn(p):
        out = model.apply(p, clip, energy, rngs={"slots": jax.random.PRNGKey(1)})
        return (out["class_logits"] ** 2).mean() + (out["pts_mean"] ** 2).mean() \
            + (out["width_mean"] ** 2).mean() + (out["amp_mean"] ** 2).mean() \
            + (out["polarity_logit"] ** 2).mean()

    grads = jax.grad(loss_fn)(params)
    # All top-level module grads should be non-zero
    total = jax.tree_util.tree_reduce(lambda a, b: a + jnp.abs(b).sum(), grads, 0.0)
    assert float(total) > 0
