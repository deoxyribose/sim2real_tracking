"""Render a set of DIVERSE-sim clips as an MP4 grid so the temporal dynamics
are visible. Layout: 8 samples, each shown as (raw | median-subtracted)."""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import jax
import matplotlib.pyplot as plt
import numpy as np

from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip

OUT = Path("/home/frans/sim2real_tracking/runs/sim_samples_video.mp4")

N_SAMPLES = 8
FPS = 8
T = 32   # more frames per clip = smoother motion


def rgb_gray(f):
    lo, hi = np.percentile(f, 1), np.percentile(f, 99)
    return np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1)


def rgb_signed(f, rng):
    return np.clip((f + rng) / (2 * rng), 0, 1)


def main():
    cfg = DiverseSimConfig(T=T, H=200, W=200)
    print(f"compiling sim (T={T})...", flush=True)

    samples = []
    for i in range(N_SAMPLES):
        out = sample_clip(jax.random.key(1000 + i), cfg)
        raw = np.asarray(out["clip_raw"])
        med = np.asarray(out["clip_median"])
        rng = max(float(np.percentile(np.abs(med), 99.5)), 0.02)
        n_c = int(out["cells"]["alive"].sum())
        n_f = int(out["flagella"]["alive"].sum())
        pip = bool(out["pipette"]["present"])
        label = f"[{i}] cells={n_c} flag={n_f} pip={'y' if pip else 'n'}"
        samples.append((raw, med, rng, label))
        print(f"  seed {i}: {label}  res.p99.5={rng:.3f}", flush=True)

    # 4 rows × 4 cols = 8 sample pairs (each pair = raw+med)
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0),
                              squeeze=False,
                              gridspec_kw=dict(wspace=0.02, hspace=0.14))
    imgs = []
    for i, (raw, med, rng, label) in enumerate(samples):
        r = i // 2
        c_base = (i % 2) * 2
        im_raw = axes[r][c_base].imshow(rgb_gray(raw[0]), cmap="gray",
                                          vmin=0, vmax=1)
        im_med = axes[r][c_base + 1].imshow(rgb_signed(med[0], rng),
                                              cmap="seismic", vmin=0, vmax=1)
        axes[r][c_base].set_title(label, fontsize=7.5)
        imgs.append((im_raw, im_med, raw, med, rng))
        for cc in (c_base, c_base + 1):
            axes[r][cc].set_xticks([]); axes[r][cc].set_yticks([])

    fig.suptitle(f"diverse sim — {N_SAMPLES} clips × {T} frames @ {FPS} fps"
                 "   (each pair: raw | median-subtracted)", fontsize=10)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(OUT), fps=FPS, codec="libx264",
                                 quality=8, macro_block_size=1)
    frame_title = fig.text(0.5, 0.005, "", ha="center", fontsize=9)
    for t in range(T):
        for im_raw, im_med, raw, med, rng in imgs:
            im_raw.set_data(rgb_gray(raw[t]))
            im_med.set_data(rgb_signed(med[t], rng))
        frame_title.set_text(f"t = {t + 1}/{T}")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(buf)
    writer.close()
    plt.close(fig)
    print(f"wrote {OUT}  ({T} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
