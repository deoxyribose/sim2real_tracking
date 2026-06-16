"""TensorBoard logging (tensorboardX) with safe scalar + video helpers."""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np


class Logger:
    def __init__(self, run_dir: str):
        os.makedirs(run_dir, exist_ok=True)
        try:
            from tensorboardX import SummaryWriter

            self.writer = SummaryWriter(run_dir)
            self.enabled = True
        except Exception:
            self.writer = None
            self.enabled = False

    def scalar(self, tag: str, value, step: int) -> None:
        if not self.enabled:
            return
        if isinstance(value, jnp.ndarray):
            value = float(value)
        self.writer.add_scalar(tag, float(value), step)

    def image(self, tag: str, img, step: int) -> None:
        if not self.enabled:
            return
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = arr[None, :, :]  # (C, H, W)
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
            arr = np.moveaxis(arr, -1, 0)
        self.writer.add_image(tag, arr, step)

    def video(self, tag: str, video, step: int, fps: int = 8) -> None:
        """`video` shape (T, H, W, C). Saved as (1, T, C, H, W)."""
        if not self.enabled:
            return
        v = np.asarray(video)
        if v.ndim == 4 and v.shape[-1] in (1, 3):
            v = np.moveaxis(v, -1, 1)  # (T, C, H, W)
            v = v[None]  # (1, T, C, H, W)
            self.writer.add_video(tag, (v * 255).astype(np.uint8), step, fps=fps)

    def close(self):
        if self.enabled:
            self.writer.close()
