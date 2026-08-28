"""Smoke tests for the CP-SAT verifier."""
import numpy as np

from sim2real.data import (
    CANONICAL_H, CANONICAL_W, FlagellumSimConfig, sample_scene,
)
from sim2real.verifier import (
    render_flagellum_candidate, render_cell_candidate,
    reconstruction_delta_per_candidate,
    build_and_solve, VerifyConfig,
)


def _fake_bg(T=16, h=96, w=96, seed=0):
    return np.random.default_rng(seed).standard_normal((T, h, w)).astype(np.float32)


def test_render_flagellum_shape():
    pts = np.array([[100, 100], [110, 100], [120, 100], [130, 100],
                    [140, 100], [150, 100], [160, 100], [170, 100]], dtype=np.float32)
    r = render_flagellum_candidate(pts, width_px=4.0, amp_signed=8.0)
    assert r.shape == (CANONICAL_H, CANONICAL_W)
    assert r.min() < 0   # darker-than-BG contribution


def test_recon_delta_prefers_correct_candidate():
    """A candidate that matches the target should get a much better (more negative) delta
    than one that doesn't."""
    H, W = 64, 64
    target = np.zeros((H, W), dtype=np.float32)
    # Put a "true" curve stamp in the target: dark line at row 32
    target[30:33, 10:50] = -8.0

    good_cand = np.zeros((H, W), dtype=np.float32); good_cand[30:33, 10:50] = -8.0
    bad_cand  = np.zeros((H, W), dtype=np.float32); bad_cand[50:53, 10:50]  = -8.0

    cands = np.stack([good_cand, bad_cand])
    deltas = reconstruction_delta_per_candidate(cands, target)
    assert deltas.shape == (2,)
    # Good candidate should reduce reconstruction error (delta < 0)
    assert deltas[0] < 0
    # Bad candidate INCREASES error (adds a dark line where there wasn't one)
    assert deltas[1] > 0


def test_solver_picks_good_candidate():
    """One good + one bad candidate → solver picks the good one."""
    H, W = CANONICAL_H, CANONICAL_W
    # Build a target with one clear flagellum + one cell
    target = np.zeros((H, W), dtype=np.float32)
    # Cell at (128, 100), radius 20
    target += render_cell_candidate(np.array([128, 100]), 20, 6.0, H, W)
    # Flagellum starting at (128, 120) (on cell boundary), going right
    pts_true = np.array([[128 + i*8*0, 120 + i*10] for i in range(8)], dtype=np.float32)
    target += render_flagellum_candidate(pts_true, 4.0, 8.0, H, W)

    # Two flagellum candidates: one matches, one wrong
    good_pts = pts_true.copy()
    bad_pts = np.array([[200 + i*5, 200 + i*5] for i in range(8)], dtype=np.float32)
    flag_cands = dict(
        renders=np.stack([
            render_flagellum_candidate(good_pts, 4.0, 8.0, H, W),
            render_flagellum_candidate(bad_pts, 4.0, 8.0, H, W),
        ]),
        attachments=np.array([[128, 120], [200, 200]], dtype=np.float32),
        pts=np.stack([good_pts, bad_pts]),
        width=np.array([4.0, 4.0], dtype=np.float32),
        amp=np.array([8.0, 8.0], dtype=np.float32),
        source_slot=np.array([0, 1], dtype=np.int32),
    )
    cell_cands = dict(
        renders=np.stack([render_cell_candidate(np.array([128, 100]), 20, 6.0, H, W)]),
        centers=np.array([[128, 100]], dtype=np.float32),
        radii=np.array([20.0], dtype=np.float32),
    )
    res = build_and_solve(flag_cands, cell_cands, target,
                          VerifyConfig(max_flagella=1, max_cells=1, require_cell_for_flag=True))
    assert res["status"] in ("OPTIMAL", "FEASIBLE")
    assert res["selected_cell_idx"] == [0]
    assert res["selected_flag_idx"] == [0]  # good, not bad


def test_solver_forbids_flag_with_no_host_cell():
    """A flagellum candidate with no compatible cell should not be selected."""
    H, W = CANONICAL_H, CANONICAL_W
    target = np.zeros((H, W), dtype=np.float32)
    pts = np.array([[200 + i*5, 200 + i*5] for i in range(8)], dtype=np.float32)
    target += render_flagellum_candidate(pts, 4.0, 8.0, H, W)
    flag_cands = dict(
        renders=np.stack([render_flagellum_candidate(pts, 4.0, 8.0, H, W)]),
        attachments=np.array([[200, 200]], dtype=np.float32),
        pts=pts[None],
        width=np.array([4.0]), amp=np.array([8.0]), source_slot=np.array([0]),
    )
    # Cell is far from attachment
    cell_cands = dict(
        renders=np.stack([render_cell_candidate(np.array([50, 50]), 15, 6.0, H, W)]),
        centers=np.array([[50, 50]], dtype=np.float32),
        radii=np.array([15.0], dtype=np.float32),
    )
    res = build_and_solve(flag_cands, cell_cands, target,
                          VerifyConfig(max_flagella=1, max_cells=1,
                                        require_cell_for_flag=True, attach_slack_px=4.0))
    assert res["selected_flag_idx"] == []   # forbidden by attachment constraint
