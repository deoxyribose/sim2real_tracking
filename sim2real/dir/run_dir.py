"""End-to-end DIR pipeline: model → hypothesis pool → ILP → tracks.

Steps:
  1. Load trained energy-UNet + PCA.
  2. For each frame chunk (T frames at a time), run N noise draws.
  3. Collect all above-threshold predictions as `Hypothesis` objects.
  4. Build ILP problem via `build_problem.build_problem`.
  5. Solve with CP-SAT.
  6. Return selected hypotheses + tracks + reconstruction diagnostics.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sim2real.model.unet_energy import (
    UNetConfig, UNetEnergy, decode_curves, sample_batched_noise, unpack_pred,
)
from sim2real.dir.build_problem import BuildConfig, Hypothesis, build_problem
from sim2real.dir.solve import SolveConfig, solve_problem


@dataclass
class DIRRunConfig:
    n_noise_draws: int = 8
    score_thresh: float = 0.1
    temperature: float = 1.0
    top_k_per_draw: int = 32          # keep only top-k highest score per draw+frame
    build: BuildConfig = None
    solve: SolveConfig = None

    def __post_init__(self):
        if self.build is None: self.build = BuildConfig()
        if self.solve is None: self.solve = SolveConfig()


def load_model_ckpt(ckpt_path: str):
    d = pickle.loads(Path(ckpt_path).read_bytes())
    params = d["params"]
    cfg_d = d["cfg_u"]
    cfg = UNetConfig(**{k: v for k, v in cfg_d.items()
                        if k in UNetConfig.__dataclass_fields__})
    return params, cfg


def load_pca_scaled(path: str):
    d = np.load(path, allow_pickle=True)
    mean = np.asarray(d["mean"])
    basis = np.asarray(d["basis"])
    sigma = np.sqrt(np.asarray(d["per_mode_var"]))[:, None, None]
    return jnp.asarray(mean), jnp.asarray(basis * sigma)


def sample_hypotheses_for_frame(
    params, model: UNetEnergy, cfg_u: UNetConfig, video: jnp.ndarray,
    key, dir_cfg: DIRRunConfig, pca_mean, pca_basis,
    frame_index: int,
) -> list[Hypothesis]:
    """Run N noise draws on ONE clip of T frames; return hypotheses assigned
    to `frame_index` (we treat the middle frame of each clip as the anchor).

    For the MVP we run one clip and take the middle frame's predictions.
    A production version would run T frames overlapping — TODO.
    """
    hypos: list[Hypothesis] = []
    for i in range(dir_cfg.n_noise_draws):
        key, k = jax.random.split(key)
        noise = sample_batched_noise(k, 1, cfg_u, temperature=dir_cfg.temperature)
        pred = model.apply(params, video, noise, train=False)
        curves = np.asarray(decode_curves(pred, cfg_u, pca_mean, pca_basis))[0]
        f = unpack_pred(pred)
        widths = np.asarray(f["width"][0])
        amps = np.asarray(f["amp"][0])
        scores = np.asarray(jax.nn.sigmoid(f["score"][0]))
        flat_c = curves.reshape(-1, curves.shape[-2], 2)
        flat_w = widths.reshape(-1)
        flat_a = amps.reshape(-1)
        flat_s = scores.reshape(-1)
        # Threshold + top-k
        keep = flat_s >= dir_cfg.score_thresh
        idxs = np.where(keep)[0]
        if len(idxs) > dir_cfg.top_k_per_draw:
            idxs = idxs[np.argsort(-flat_s[idxs])[:dir_cfg.top_k_per_draw]]
        for j in idxs:
            hypos.append(Hypothesis(
                frame=frame_index,
                skeleton=flat_c[j].astype(np.float32),
                width=float(flat_w[j]),
                amp=float(flat_a[j]),
                score=float(flat_s[j]),
            ))
    return hypos


def run_dir_on_clips(
    ckpt_path: str, pca_path: str,
    clips: np.ndarray,                    # (T_total, H, W) residual sequence
    dir_cfg: DIRRunConfig = None,
) -> dict:
    """End-to-end pipeline for a residual clip sequence.

    For MVP: split the sequence into non-overlapping windows of `cfg_u.T`
    frames; for each window run n noise draws on the WHOLE window and use
    the middle frame's predictions as one frame's hypothesis pool. Result:
    one frame's worth of hypotheses per window. Suitable for short clips.
    """
    if dir_cfg is None:
        dir_cfg = DIRRunConfig()

    params, cfg_u = load_model_ckpt(ckpt_path)
    pca_mean, pca_basis = load_pca_scaled(pca_path)
    model = UNetEnergy(cfg=cfg_u)

    T_total, H, W = clips.shape
    assert H == cfg_u.H and W == cfg_u.W, \
        f"clip shape {H,W} doesn't match model {cfg_u.H,cfg_u.W}"

    # For a first prototype: slide a T-window with stride T//2, use the middle
    # frame of each window as an anchor.
    stride = max(1, cfg_u.T // 2)
    windows = []
    anchors = []
    t = 0
    while t + cfg_u.T <= T_total:
        windows.append(clips[t : t + cfg_u.T])
        anchors.append(t + cfg_u.T // 2)
        t += stride
    if not windows:
        # pad the whole thing to T frames
        pad = cfg_u.T - T_total
        padded = np.concatenate([clips, np.zeros((pad, H, W), np.float32)], 0)
        windows = [padded]
        anchors = [T_total // 2]

    key = jax.random.key(0)
    hypotheses: list[Hypothesis] = []
    for win_idx, (win, anchor) in enumerate(zip(windows, anchors)):
        video = jnp.asarray(win)[None]                        # (1, T, H, W)
        key, kk = jax.random.split(key)
        hs = sample_hypotheses_for_frame(params, model, cfg_u, video, kk,
                                          dir_cfg, pca_mean, pca_basis,
                                          frame_index=anchor)
        hypotheses.extend(hs)

    # Reconstruction target: use ALL anchor frames' residuals as the "target".
    # Since we assign frame index = anchor, we need residual[anchor] for each.
    # For simplicity pack per-anchor residuals into a stacked (T_anchor, H, W).
    unique_anchors = sorted(set(anchors))
    anchor_to_idx = {a: i for i, a in enumerate(unique_anchors)}
    per_anchor_res = np.stack([clips[a] for a in unique_anchors], axis=0)
    # Rewrite each hypothesis.frame to its ANCHOR-index
    for h in hypotheses:
        h.frame = anchor_to_idx[h.frame]

    problem = build_problem(hypotheses, per_anchor_res, dir_cfg.build)
    print(f"DIR problem: {problem['num_variables']} vars, "
          f"{len(problem['at_most_one_constraints'])} amo, "
          f"{len(problem['links'])} links", flush=True)

    solution = solve_problem(problem, dir_cfg.solve)
    print(f"DIR solved: status={solution['status']}  obj={solution['objective']:.1f}  "
          f"selected={len(solution['selected_indices'])}  tracks={len(solution['tracks'])}"
          f"  t={solution['wall_time']:.2f}s", flush=True)

    return dict(
        hypotheses=hypotheses,
        problem=problem,
        solution=solution,
        anchor_frames=unique_anchors,
    )
