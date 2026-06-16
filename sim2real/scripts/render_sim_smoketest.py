"""Render a few sample videos from every simulator and dump them as PNG montages + mp4 if ffmpeg
is available. Intended as a fast eyeball check that the sim primitives are correctly composing.

Usage:
    PYTHONPATH=. python3 -m sim2real.scripts.render_sim_smoketest --out runs/sim_smoke
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

from sim2real.sim.api import build_sim, list_sims


def _save_montage(video, masks, out_png: str):
    """Save first 4 frames + their composite-with-mask-overlay as a single PNG."""
    import matplotlib.pyplot as plt

    T = min(4, video.shape[0])
    n_act = int((masks[0].sum(axis=(-1, -2)) > 0.01).sum())  # rough alive count

    fig, axes = plt.subplots(2, T, figsize=(3 * T, 6))
    for t in range(T):
        axes[0, t].imshow(np.asarray(video[t, ..., 0]), cmap="gray", vmin=0, vmax=1)
        axes[0, t].set_title(f"frame {t}")
        axes[0, t].axis("off")
        # Combined mask (sum over slots, clipped)
        m_sum = np.clip(np.asarray(masks[t]).sum(0), 0, 1)
        axes[1, t].imshow(np.asarray(video[t, ..., 0]), cmap="gray", vmin=0, vmax=1)
        axes[1, t].imshow(m_sum, cmap="Reds", alpha=0.45)
        axes[1, t].set_title(f"masks (n≈{n_act})")
        axes[1, t].axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=90)
    plt.close(fig)


def _save_mp4(video, out_mp4: str):
    """Save the full video as mp4 via imageio. Skip silently if no ffmpeg."""
    try:
        import imageio.v2 as imageio

        frames = (np.asarray(video[..., 0]) * 255).astype(np.uint8)
        imageio.mimwrite(out_mp4, frames, fps=8)
    except Exception as e:  # noqa: BLE001
        print(f"[mp4 skipped] {e}")


def main(out_dir: str, batch: int, seed: int):
    os.makedirs(out_dir, exist_ok=True)
    for kind in list_sims():
        print(f"-- {kind}")
        batch_fn, cfg = build_sim(kind)
        key = jax.random.key(seed)
        sample = jax.jit(lambda k: batch_fn(k, batch))(key)
        for i in range(batch):
            video_i = sample.video[i]                                 # (T, H, W, 1)
            masks_i = sample.masks[i]                                 # (T, N, H, W)
            stem = os.path.join(out_dir, f"{kind}_{i:02d}")
            _save_montage(video_i, masks_i, stem + ".png")
            _save_mp4(video_i, stem + ".mp4")
        print(f"   wrote {batch} samples to {out_dir}/{kind}_*.png")
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/sim_smoke")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.out, args.batch, args.seed)
