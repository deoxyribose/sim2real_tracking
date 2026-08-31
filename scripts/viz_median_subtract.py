"""Visualize two real videos with (a) raw, (b) global temporal-median subtracted,
(c) sliding-window temporal-median subtracted.

Writes both a still (PNG) and a 3-panel-per-video (MP4).
"""
from __future__ import annotations

import glob
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path("/home/frans/sim2real_tracking/algae_data/CC124_pipette_wt_beat")

VIDEOS = [
    # (label, dir, glob, crop_bottom): frames are cropped to [0:crop_bottom]
    # to strip the burnt-in timestamp banner (text on white background).
    ("cell2 / Export_20211015_124200",
     ROOT / "CC124_cell2/Export_20211015_124200", "*.bmp", 200),
    ("cell1 / Export_20211007_174615",
     ROOT / "CC124_cell1/2021_10_07/Export_20211007_174615", "*.bmp", 176),
]

N_FRAMES = 240          # frames to load per video (after stride)
FRAME_STRIDE = 3        # load every k-th file: spans ~N_FRAMES*STRIDE raw frames
WINDOW = 21             # sliding-window size (frames) for local median
FPS = 20                # output video framerate


def load_stack(dirpath: Path, pattern: str, n_frames: int,
               stride: int, crop_bottom: int) -> np.ndarray:
    files = sorted(glob.glob(str(dirpath / pattern)))
    if not files:
        raise FileNotFoundError(f"no frames in {dirpath}")
    start = max(0, (len(files) - n_frames * stride) // 4)
    files = files[start : start + n_frames * stride : stride]
    frames = []
    for f in files:
        img = np.array(Image.open(f))
        if img.ndim == 3:
            img = img.mean(axis=-1)
        frames.append(img[:crop_bottom].astype(np.float32))
    return np.stack(frames)  # (T, H, W)


def sliding_median(stack: np.ndarray, window: int) -> np.ndarray:
    T = stack.shape[0]
    half = window // 2
    out = np.empty_like(stack)
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        out[t] = np.median(stack[lo:hi], axis=0)
    return out


def make_still(video_data, out_path: Path):
    """One representative frame per video (2 rows x 3 cols)."""
    n = len(video_data)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3.2 * n), squeeze=False)
    col_titles = ["raw frame", "− global median", "− window median"]

    for row, (label, stack, gmed, wmed, rng) in enumerate(video_data):
        t = stack.shape[0] // 2
        axes[row][0].imshow(stack[t], cmap="gray", vmin=0, vmax=255)
        axes[row][1].imshow(stack[t] - gmed, cmap="seismic", vmin=-rng, vmax=rng)
        axes[row][2].imshow(stack[t] - wmed[t], cmap="seismic", vmin=-rng, vmax=rng)
        axes[row][0].set_ylabel(label, fontsize=9)
        for c in range(3):
            axes[row][c].set_xticks([])
            axes[row][c].set_yticks([])
            if row == 0:
                axes[row][c].set_title(col_titles[c], fontsize=10)

    fig.suptitle(f"temporal-median subtraction  (window = {WINDOW} frames)",
                 fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def make_video(video_data, out_path: Path, fps: int):
    """2 rows x 3 cols animation, one frame per input timestep."""
    T = video_data[0][1].shape[0]
    n = len(video_data)
    col_titles = ["raw frame", "− global median", "− window median"]

    fig, axes = plt.subplots(n, 3, figsize=(9, 3.2 * n), squeeze=False)
    imgs = []  # keep AxesImage handles to update
    for row, (label, stack, gmed, wmed, rng) in enumerate(video_data):
        im0 = axes[row][0].imshow(stack[0], cmap="gray", vmin=0, vmax=255)
        im1 = axes[row][1].imshow(stack[0] - gmed, cmap="seismic",
                                   vmin=-rng, vmax=rng)
        im2 = axes[row][2].imshow(stack[0] - wmed[0], cmap="seismic",
                                   vmin=-rng, vmax=rng)
        imgs.append((im0, im1, im2))
        axes[row][0].set_ylabel(label, fontsize=9)
        for c in range(3):
            axes[row][c].set_xticks([])
            axes[row][c].set_yticks([])
            if row == 0:
                axes[row][c].set_title(col_titles[c], fontsize=10)
    title = fig.suptitle("", fontsize=11)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                quality=8, macro_block_size=1)
    for t in range(T):
        title.set_text(f"temporal-median subtraction  window={WINDOW}   "
                       f"t={t + 1}/{T}")
        for row, (label, stack, gmed, wmed, rng) in enumerate(video_data):
            imgs[row][0].set_data(stack[t])
            imgs[row][1].set_data(stack[t] - gmed)
            imgs[row][2].set_data(stack[t] - wmed[t])
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(frame)
    writer.close()
    plt.close(fig)
    print(f"wrote {out_path}  ({T} frames @ {fps} fps)")


def main():
    video_data = []
    for label, dirpath, pattern, crop_bottom in VIDEOS:
        print(f"loading {label} ...")
        stack = load_stack(dirpath, pattern, N_FRAMES, FRAME_STRIDE, crop_bottom)
        print(f"  stack {stack.shape} range [{stack.min():.0f}, {stack.max():.0f}]")
        gmed = np.median(stack, axis=0)
        wmed = sliding_median(stack, WINDOW)
        # robust common colour scale across both subtractions for this video
        diffs = np.concatenate([(stack - gmed).ravel(),
                                (stack - wmed).ravel()])
        rng = float(max(np.percentile(np.abs(diffs), 99), 1.0))
        video_data.append((label, stack, gmed, wmed, rng))

    out_dir = Path("/home/frans/sim2real_tracking/runs")
    make_still(video_data, out_dir / "viz_median_subtract.png")
    make_video(video_data, out_dir / "viz_median_subtract.mp4", FPS)


if __name__ == "__main__":
    main()
