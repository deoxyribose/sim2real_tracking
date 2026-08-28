"""Tests for the extended Hungarian + NLL loss (cells + flagella)."""
import jax
import jax.numpy as jnp
import numpy as np

from sim2real.data import (
    CANONICAL_H, CANONICAL_W, CLASS_EMPTY, CLASS_CELL, CLASS_FLAGELLUM,
    FLAGELLUM_K, N_CLASSES,
    CellLatent, FlagellumLatent, SceneLatents,
)
from sim2real.model_v2 import (
    MAX_GT_FLAG, MAX_GT_CELL,
    DETRSlotConfig, DETRSlotModel,
    compute_loss, hungarian_match_flagella, hungarian_match_cells, pack_gt_batch, run_matches,
)


def _make_scene(n_cells=1, n_flag=1, seed=0):
    rng = np.random.default_rng(seed)
    cells = []
    flagella = []
    for _ in range(n_cells):
        c = CellLatent(center=rng.uniform(80, 180, size=2).astype(np.float32),
                       radius_px=float(rng.uniform(15, 30)),
                       amplitude_sigma=float(rng.uniform(4, 10)))
        cells.append(c)
    for _ in range(n_flag):
        attach = rng.uniform(50, 200, size=2).astype(np.float32)
        angle = rng.uniform(0, 2*np.pi)
        step = 8
        pts = np.stack([attach + i*step*np.array([np.sin(angle), np.cos(angle)])
                        for i in range(1, FLAGELLUM_K + 1)]).astype(np.float32)
        flagella.append(FlagellumLatent(
            attachment=attach, control_points=pts,
            width_px=4.0, polarity=1, amplitude_sigma=8.0,
        ))
    return SceneLatents(flagella=flagella, cells=cells)


def test_pack_gt_batch_shapes():
    scenes = [_make_scene(1, 1), _make_scene(2, 3), _make_scene(0, 0)]
    gt = pack_gt_batch(scenes)
    assert gt["gt_pts"].shape == (3, MAX_GT_FLAG, FLAGELLUM_K + 1, 2)
    assert gt["gt_cell_center"].shape == (3, MAX_GT_CELL, 2)
    # Second scene has 3 flagella but MAX_GT_FLAG=4, all fit
    assert gt["gt_mask"][0].tolist() == [1, 0, 0, 0]
    assert gt["gt_mask"][1].tolist() == [1, 1, 1, 0]
    assert gt["gt_mask"][2].tolist() == [0, 0, 0, 0]
    assert gt["gt_cell_mask"][0].tolist() == [1, 0]
    assert gt["gt_cell_mask"][1].tolist() == [1, 1]


def test_hungarian_cells_matches_close():
    scene = _make_scene(1, 0)
    gt = pack_gt_batch([scene])
    S = 4
    class_logits = np.full((1, S, N_CLASSES), -5.0, dtype=np.float32)
    class_logits[0, 1, CLASS_CELL] = 5.0
    pred_center = np.full((1, S, 2), 300.0, dtype=np.float32)
    pred_center[0, 1] = gt["gt_cell_center"][0, 0]
    pred_radius = np.full((1, S), 30.0, dtype=np.float32)
    pred_radius[0, 1] = gt["gt_cell_radius"][0, 0]
    match = hungarian_match_cells(class_logits, pred_center, pred_radius,
                                   gt["gt_cell_center"], gt["gt_cell_radius"], gt["gt_cell_mask"])
    assert match[0, 1] == 0
    assert (match[0] == -1).sum() == S - 1


def test_hungarian_flag_avoids_cell_slots():
    scene = _make_scene(1, 1)
    gt = pack_gt_batch([scene])
    S = 4
    class_logits = np.full((1, S, N_CLASSES), -5.0, dtype=np.float32)
    class_logits[0, :, CLASS_FLAGELLUM] = 2.0
    pred_pts = np.full((1, S, FLAGELLUM_K + 1, 2), 100.0, dtype=np.float32)
    pred_pts[0, 0] = gt["gt_pts"][0, 0]   # slot 0 predicts GT flag exactly
    forbidden = np.array([[False, True, False, False]])   # slot 1 already used by cell
    match = hungarian_match_flagella(class_logits, pred_pts, gt["gt_pts"], gt["gt_mask"],
                                      slot_forbidden_np=forbidden)
    assert match[0, 0] == 0
    assert match[0, 1] == -1


def test_compute_loss_smoke_and_grad():
    B = 2
    scenes = [_make_scene(1, 1, seed=0), _make_scene(2, 3, seed=1)]
    gt_np = pack_gt_batch(scenes)
    gt = {k: jnp.asarray(v) for k, v in gt_np.items()}

    cfg = DETRSlotConfig(n_slots=6)
    model = DETRSlotModel(cfg=cfg)
    T = 8
    clip = jnp.zeros((B, T, CANONICAL_H, CANONICAL_W))
    energy = jnp.zeros((B, CANONICAL_H, CANONICAL_W))
    key = jax.random.PRNGKey(0)
    params = model.init({"params": key, "slots": key}, clip, energy)

    out_fwd = model.apply(params, clip, energy, rngs={"slots": key})
    match_flag, match_cell = run_matches({k: np.asarray(v) for k, v in out_fwd.items()}, gt_np)
    match_flag_j = jnp.asarray(match_flag)
    match_cell_j = jnp.asarray(match_cell)

    def loss_fn(p):
        out = model.apply(p, clip, energy, rngs={"slots": key})
        losses = compute_loss(out, gt, match_flag_j, match_cell_j)
        return losses["total"], losses

    (total, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    assert jnp.isfinite(total)
    for k in ("class_", "pts", "width", "amp", "polarity", "cell_pos", "cell_rad", "cell_amp"):
        assert jnp.isfinite(losses[k]), f"{k}={losses[k]}"
    total_grad = jax.tree_util.tree_reduce(lambda a, b: a + jnp.abs(b).sum(), grads, 0.0)
    assert float(total_grad) > 0
