"""Run the full trained pipeline on a real video sequence and emit a demo
MP4 with tracked skeleton overlays.

For each sliding window of T frames:
  - canonicalize to residual space
  - run model with N noise draws
  - collect middle-frame candidates
Then feed all anchor frames to DIR with temporal linking, so selection is
coherent across time.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image

from sim2real.data.canonicalize import CanonicalConfig, canonicalize_clip
from sim2real.dir.build_problem import BuildConfig, Hypothesis, build_problem
from sim2real.dir.run_dir import load_model_ckpt, load_pca_scaled
from sim2real.dir.solve import SolveConfig, solve_problem
from sim2real.model.unet_energy import (
    decode_curves, sample_batched_noise, UNetEnergy, unpack_pred,
)


def load_frame(path: str) -> np.ndarray:
    if path.endswith(".tif"):
        img = tifffile.imread(path)
    else:
        img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img.mean(-1)
    return img.astype(np.float32)


def load_sequence(dir_path: Path, ext: str, crop_bottom: int,
                  n_frames: int, start: int = 0) -> np.ndarray:
    files = sorted(glob.glob(str(dir_path / f"*{ext}")))[start : start + n_frames]
    frames = [load_frame(f)[:crop_bottom] for f in files]
    return np.stack(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pca", default="data_cache/flagella_pca.npz")
    ap.add_argument("--seq-dir", required=True, help="directory with the raw frames")
    ap.add_argument("--ext", default=".bmp")
    ap.add_argument("--crop-bottom", type=int, default=200,
                    help="strip the burnt-in timestamp banner")
    ap.add_argument("--src-width", type=float, default=4.0,
                    help="median flagellum width in native px — canonicalize resample")
    ap.add_argument("--n-frames", type=int, default=64,
                    help="number of frames from `seq-dir` to process")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n-draws", type=int, default=6)
    ap.add_argument("--score-thresh", type=float, default=0.03)
    ap.add_argument("--top-k", type=int, default=24)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params, cfg_u = load_model_ckpt(args.ckpt)
    pca_mean, pca_basis = load_pca_scaled(args.pca)
    model = UNetEnergy(cfg=cfg_u)

    print(f"loading {args.n_frames} frames from {args.seq_dir}", flush=True)
    stack = load_sequence(Path(args.seq_dir), args.ext, args.crop_bottom,
                          args.n_frames, args.start) / 255.0

    print(f"canonicalising to residual (T={stack.shape[0]}, {stack.shape[1]}x{stack.shape[2]})...", flush=True)
    canon = canonicalize_clip(stack, CanonicalConfig(src_width_px=args.src_width,
                                                        bg_median_window=15))
    residual = canon["clip"]                            # (T, H, W) canonical
    H_c, W_c = residual.shape[1:]
    # Pad to model input size if needed
    if H_c != cfg_u.H or W_c != cfg_u.W:
        ph = cfg_u.H - H_c; pw = cfg_u.W - W_c
        if ph < 0 or pw < 0:
            print(f"canonical clip too big for model: {residual.shape}"); return
        residual = np.pad(residual, ((0, 0), (0, ph), (0, pw)), constant_values=0.0)

    # Run model on sliding windows to get candidates per anchor frame
    T_total = residual.shape[0]
    stride = max(1, cfg_u.T // 4)
    anchors = []
    per_anchor_cands: dict[int, list[Hypothesis]] = {}
    key = jax.random.key(0)
    for t0 in range(0, T_total - cfg_u.T + 1, stride):
        anchor = t0 + cfg_u.T // 2
        anchors.append(anchor)
        win = residual[t0 : t0 + cfg_u.T]
        video = jnp.asarray(win)[None]
        cands: list[Hypothesis] = []
        for _ in range(args.n_draws):
            key, k = jax.random.split(key)
            noise = sample_batched_noise(k, 1, cfg_u)
            pred = model.apply(params, video, noise, train=False)
            curves = np.asarray(decode_curves(pred, cfg_u, pca_mean, pca_basis))[0]
            f = unpack_pred(pred)
            w = np.asarray(f["width"][0]); a = np.asarray(f["amp"][0])
            s = np.asarray(jax.nn.sigmoid(f["score"][0]))
            flat_c = curves.reshape(-1, curves.shape[-2], 2)
            flat_w = w.reshape(-1); flat_a = a.reshape(-1); flat_s = s.reshape(-1)
            keep = np.where(flat_s >= args.score_thresh)[0]
            if len(keep) > args.top_k:
                keep = keep[np.argsort(-flat_s[keep])[: args.top_k]]
            for j in keep:
                cands.append(Hypothesis(
                    frame=anchor,      # anchor time index (into `residual`)
                    skeleton=flat_c[j].astype(np.float32),
                    width=float(flat_w[j]), amp=float(flat_a[j]),
                    score=float(flat_s[j])))
        per_anchor_cands[anchor] = cands
    print(f"n_anchors={len(anchors)}   total_candidates="
          f"{sum(len(v) for v in per_anchor_cands.values())}", flush=True)

    # Assemble full hypothesis list with anchor→index mapping for DIR
    anchor_list = sorted(per_anchor_cands.keys())
    a2i = {a: i for i, a in enumerate(anchor_list)}
    hypos = []
    for a in anchor_list:
        for h in per_anchor_cands[a]:
            h.frame = a2i[a]
            hypos.append(h)
    per_anchor_res = np.stack([residual[a] for a in anchor_list], axis=0)

    # Tuned to give a small handful of picks per frame: score-only cost with
    # a non-trivial pick_cost_base so only the higher-score candidates qualify,
    # plus small birth/death to discourage 1-frame tracks.
    build_cfg = BuildConfig(cost_mode="score_only", pick_cost_base=5.0,
                              score_bonus=100.0, birth_cost=3.0, death_cost=3.0,
                              link_max_dist=25.0)
    problem = build_problem(hypos, per_anchor_res, build_cfg)
    print(f"DIR problem: {problem['num_variables']} vars, "
          f"{len(problem['at_most_one_constraints'])} amo, "
          f"{len(problem['links'])} links", flush=True)
    sol = solve_problem(problem, SolveConfig(time_limit_s=60.0))
    print(f"DIR solved: obj={sol['objective']:.1f}  sel={len(sol['selected_indices'])}  "
          f"tracks={len(sol['tracks'])}", flush=True)

    # Render an MP4: raw frame + selected skeletons per frame
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(np.zeros((cfg_u.H, cfg_u.W)), cmap="gray")
    ttl = ax.set_title("", fontsize=10)
    fig.tight_layout()

    # Group selected hypos by anchor
    hypos_by_anchor: dict[int, list[Hypothesis]] = {}
    for si in sol["selected_indices"]:
        h = hypos[si]
        hypos_by_anchor.setdefault(h.frame, []).append(h)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.out), fps=args.fps, codec="libx264",
                                 quality=8, macro_block_size=1)
    for i, a in enumerate(anchor_list):
        frame = residual[a]
        rng = max(float(np.percentile(np.abs(frame), 99.5)), 0.02)
        ax.clear()
        ax.imshow(np.clip((frame + rng) / (2 * rng), 0, 1), cmap="seismic")
        for h in hypos_by_anchor.get(i, []):
            ax.plot(h.skeleton[:, 1], h.skeleton[:, 0], "-", color="#ff9500",
                    linewidth=1.6, alpha=0.85)
        ax.set_title(f"t={a}  n_sel={len(hypos_by_anchor.get(i, []))}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, cfg_u.W - 1); ax.set_ylim(cfg_u.H - 1, 0)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(buf)
    writer.close()
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
