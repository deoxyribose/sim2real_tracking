"""Load a trained energy-U-Net checkpoint, run several forward passes with
different noise seeds on the SAME sim clip, and overlay all candidate
skeletons on the input frame."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from sim2real.model.unet_energy import (
    UNetConfig, UNetEnergy, decode_curves, sample_batched_noise,
)
from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip


def load_ckpt(path: str) -> tuple[dict, UNetConfig]:
    d = pickle.loads(Path(path).read_bytes())
    params = d["params"]
    cfg_d = d["cfg_u"]
    # Recreate the config; support both older/newer field sets
    cfg = UNetConfig(**{k: v for k, v in cfg_d.items()
                        if k in UNetConfig.__dataclass_fields__})
    return params, cfg


def load_pca(path: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    d = np.load(path, allow_pickle=True)
    mean = np.asarray(d["mean"])
    basis = np.asarray(d["basis"])
    sigma = np.sqrt(np.asarray(d["per_mode_var"]))[:, None, None]
    return jnp.asarray(mean), jnp.asarray(basis * sigma)


def rgb_gray(f):
    lo, hi = np.percentile(f, 1), np.percentile(f, 99)
    return np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1)


def draw_candidates(ax, curves, scores, top_k, color, alpha=0.5,
                    linewidth=1.1):
    """Overlay the top-k candidate skeletons by score."""
    flat_c = curves.reshape(-1, curves.shape[-2], 2)
    flat_s = scores.reshape(-1)
    order = np.argsort(-flat_s)[:top_k]
    for i in order:
        ax.plot(flat_c[i, :, 1], flat_c[i, :, 0], "-", color=color,
                linewidth=linewidth, alpha=alpha)
    return len(order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/energy_v0/ckpt_step005000.pkl")
    ap.add_argument("--pca", default="data_cache/flagella_pca.npz")
    ap.add_argument("--n-samples", type=int, default=8, help="noise seeds per clip")
    ap.add_argument("--n-clips", type=int, default=4, help="sim clips to viz")
    ap.add_argument("--top-k", type=int, default=20,
                    help="top-k highest-score candidates per noise draw to overlay")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--out", default="runs/energy_v0/preds.png")
    args = ap.parse_args()

    params, cfg_u = load_ckpt(args.ckpt)
    pca_mean, pca_basis = load_pca(args.pca)
    model = UNetEnergy(cfg=cfg_u)
    sim_cfg = DiverseSimConfig(T=cfg_u.T, H=cfg_u.H, W=cfg_u.W)
    print(f"loaded ckpt (grid {cfg_u.grid_h}x{cfg_u.grid_w}, "
          f"{cfg_u.n_suggestions} suggestions/cell)")

    key = jax.random.key(args.seed)
    fig, axes = plt.subplots(args.n_clips, 3, figsize=(15, 4.5 * args.n_clips),
                              squeeze=False)

    for r in range(args.n_clips):
        key, k_clip = jax.random.split(key)
        out = sample_clip(k_clip, sim_cfg)
        raw = np.asarray(out["clip_raw"])
        med = np.asarray(out["clip_median"])
        gt = np.asarray(out["curves"])[cfg_u.T // 2]           # (N_flag, K, 2)
        alive = np.asarray(out["flagella"]["alive"])
        gt_valid = [gt[k] for k in range(gt.shape[0]) if bool(alive[k])]

        video = med[None]                                       # (1, T, H, W)
        # Multiple noise samples over the SAME clip
        all_curves = []
        all_scores = []
        for i in range(args.n_samples):
            key, k_n = jax.random.split(key)
            noise = sample_batched_noise(k_n, 1, cfg_u, temperature=args.temp)
            pred = model.apply(params, jnp.asarray(video), noise, train=False)
            curves = np.asarray(decode_curves(pred, cfg_u, pca_mean, pca_basis))[0]
            scores = np.asarray(jax.nn.sigmoid(pred[..., -1]))[0]
            all_curves.append(curves); all_scores.append(scores)

        t = raw.shape[0] // 2
        rng = max(float(np.percentile(np.abs(med[t]), 99.5)), 0.02)

        # Panel 0: raw + GT
        axes[r][0].imshow(rgb_gray(raw[t]), cmap="gray")
        for g in gt_valid:
            axes[r][0].plot(g[:, 1], g[:, 0], "-", color="#33dd33", linewidth=2.5)
        axes[r][0].set_title(f"clip {r}   raw + GT (green)", fontsize=10)

        # Panel 1: median-subtracted (model input) + GT
        axes[r][1].imshow(np.clip((med[t] + rng) / (2 * rng), 0, 1), cmap="seismic")
        for g in gt_valid:
            axes[r][1].plot(g[:, 1], g[:, 0], "-", color="#33dd33", linewidth=2.5)
        axes[r][1].set_title(f"median-subtracted input + GT", fontsize=10)

        # Panel 2: raw + N samples of candidate overlay
        axes[r][2].imshow(rgb_gray(raw[t]), cmap="gray")
        colors = plt.cm.viridis(np.linspace(0.1, 0.95, args.n_samples))
        total = 0
        for i in range(args.n_samples):
            total += draw_candidates(axes[r][2], all_curves[i], all_scores[i],
                                       top_k=args.top_k,
                                       color=tuple(colors[i]),
                                       alpha=0.35, linewidth=1.0)
        # overlay GT on top
        for g in gt_valid:
            axes[r][2].plot(g[:, 1], g[:, 0], "-", color="#33dd33", linewidth=2.5)
        axes[r][2].set_title(
            f"{args.n_samples} noise draws  x  top-{args.top_k} = {total} candidates",
            fontsize=10)

        for c in range(3):
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
            axes[r][c].set_xlim(0, cfg_u.W - 1); axes[r][c].set_ylim(cfg_u.H - 1, 0)

    fig.suptitle(f"energy-UNet predictions   —   ckpt {args.ckpt.split('/')[-1]}   "
                 f"temperature={args.temp}", fontsize=11)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
