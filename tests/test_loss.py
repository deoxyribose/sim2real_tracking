"""Tests for the Hungarian + NLL loss."""
import jax
import jax.numpy as jnp
import numpy as np

from sim2real.data import (
    CANONICAL_H, CANONICAL_W, CLASS_EMPTY, CLASS_FLAGELLUM, FLAGELLUM_K, N_CLASSES,
    FlagellumLatent, SceneLatents,
)
from sim2real.model_v2 import (
    MAX_GT, DETRSlotConfig, DETRSlotModel, compute_loss, hungarian_match, pack_gt_batch,
)


def _make_scene(n_flag=1, seed=0):
    rng = np.random.default_rng(seed)
    flagella = []
    for _ in range(n_flag):
        attach = rng.uniform(50, 200, size=2).astype(np.float32)
        angles = rng.uniform(0, 2*np.pi)
        step = 8
        pts = np.stack([attach + i*step*np.array([np.sin(angles), np.cos(angles)]) for i in range(1, FLAGELLUM_K+1)]).astype(np.float32)
        flagella.append(FlagellumLatent(
            attachment=attach, control_points=pts,
            width_px=4.0, polarity=1, amplitude_sigma=8.0,
        ))
    return SceneLatents(flagella=flagella)


def test_pack_gt_batch_shapes():
    scenes = [_make_scene(1), _make_scene(2), _make_scene(0)]
    gt = pack_gt_batch(scenes)
    assert gt["gt_pts"].shape == (3, MAX_GT, FLAGELLUM_K + 1, 2)
    assert gt["gt_mask"].tolist() == [[1, 0], [1, 1], [0, 0]]


def test_hungarian_match_matches_close():
    """When the model puts a slot exactly at GT, that slot should match."""
    scene = _make_scene(1)
    gt = pack_gt_batch([scene])
    S = 4
    class_logits = np.full((1, S, N_CLASSES), -5.0, dtype=np.float32)
    class_logits[0, 2, CLASS_FLAGELLUM] = 5.0   # slot 2 says "I'm flagellum!"
    pred_pts = np.full((1, S, FLAGELLUM_K + 1, 2), 200.0, dtype=np.float32)
    pred_pts[0, 2] = gt["gt_pts"][0, 0]         # slot 2 predicts GT exactly
    match = hungarian_match(class_logits, pred_pts, gt["gt_pts"], gt["gt_mask"])
    assert match.shape == (1, S)
    assert match[0, 2] == 0
    assert (match[0] == -1).sum() == S - 1


def test_hungarian_no_match_when_no_gt():
    scene = _make_scene(0)
    gt = pack_gt_batch([scene])
    S = 4
    pred_pts = np.random.randn(1, S, FLAGELLUM_K + 1, 2).astype(np.float32) * 5 + 100
    class_logits = np.zeros((1, S, N_CLASSES), dtype=np.float32)
    match = hungarian_match(class_logits, pred_pts, gt["gt_pts"], gt["gt_mask"])
    assert (match == -1).all()


def test_compute_loss_smoke_and_grad():
    B = 2
    scenes = [_make_scene(1, seed=0), _make_scene(2, seed=1)]
    gt_np = pack_gt_batch(scenes)
    gt = {k: jnp.asarray(v) for k, v in gt_np.items()}

    cfg = DETRSlotConfig(n_slots=6)
    model = DETRSlotModel(cfg=cfg)
    T = 8
    clip = jnp.zeros((B, T, CANONICAL_H, CANONICAL_W))
    energy = jnp.zeros((B, CANONICAL_H, CANONICAL_W))
    key = jax.random.PRNGKey(0)
    params = model.init({"params": key, "slots": key}, clip, energy)

    # Step 1: run forward on host, get outputs as numpy for matching
    out_fwd = model.apply(params, clip, energy, rngs={"slots": key})
    match_np = hungarian_match(
        np.asarray(out_fwd["class_logits"]),
        np.asarray(out_fwd["pts_mean"]),
        gt_np["gt_pts"], gt_np["gt_mask"],
    )
    match = jnp.asarray(match_np)

    # Step 2: define loss_fn that takes params and pre-computed match
    def loss_fn(p):
        out = model.apply(p, clip, energy, rngs={"slots": key})
        losses = compute_loss(out, gt, match)
        return losses["total"], losses

    (total, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    assert jnp.isfinite(total)
    # Component sanity
    for k in ("class_", "pts", "width", "amp", "polarity"):
        assert jnp.isfinite(losses[k]), f"{k}={losses[k]}"
    # Non-zero gradients
    total_grad = jax.tree_util.tree_reduce(lambda a, b: a + jnp.abs(b).sum(), grads, 0.0)
    assert float(total_grad) > 0
