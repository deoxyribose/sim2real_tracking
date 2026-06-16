"""Visualization helpers for inspecting model outputs."""

from __future__ import annotations

import numpy as np


def composite_overlay(video, masks, alpha: float = 0.45):
    """Compose per-frame video with mask overlay. video: (T, H, W, C); masks: (T, N, H, W)."""
    import matplotlib.cm as cm

    T = video.shape[0]
    fig_rows = 2
    fig_cols = min(T, 4)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(3 * fig_cols, 6))
    if fig_cols == 1:
        axes = axes.reshape(2, 1)
    for t in range(fig_cols):
        axes[0, t].imshow(np.asarray(video[t, ..., 0]), cmap="gray", vmin=0, vmax=1)
        axes[0, t].axis("off")
        m_sum = np.clip(np.asarray(masks[t]).sum(0), 0, 1)
        axes[1, t].imshow(np.asarray(video[t, ..., 0]), cmap="gray", vmin=0, vmax=1)
        axes[1, t].imshow(m_sum, cmap="Reds", alpha=alpha)
        axes[1, t].axis("off")
    fig.tight_layout()
    return fig
