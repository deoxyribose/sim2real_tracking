"""Sanity-check the fitted PCA: show the mean curve + top 8 modes at ±3σ."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PCA = "/home/frans/sim2real_tracking/data_cache/flagella_pca.npz"
OUT = Path("/home/frans/sim2real_tracking/runs/flagella_pca_modes.png")


def main():
    d = np.load(PCA, allow_pickle=True)
    mean = d["mean"]                     # (K, 2)
    basis = d["basis"]                   # (M, K, 2)
    per_mode_var = d["per_mode_var"]     # (M,)

    M = min(basis.shape[0], 8)
    fig, axes = plt.subplots(1, M + 1, figsize=(2.6 * (M + 1), 3.0))
    axes[0].plot(mean[:, 1], mean[:, 0], "k-", linewidth=2, label="mean")
    axes[0].set_title("mean canonical curve", fontsize=9)
    axes[0].axis("equal")
    axes[0].axhline(0, color="#ccc", lw=0.5); axes[0].axvline(0, color="#ccc", lw=0.5)

    for i in range(M):
        sigma = float(np.sqrt(per_mode_var[i]))
        plus = mean + 3 * sigma * basis[i]
        minus = mean - 3 * sigma * basis[i]
        axes[i + 1].plot(mean[:, 1], mean[:, 0], "k--", linewidth=1.0, alpha=0.5)
        axes[i + 1].plot(plus[:, 1], plus[:, 0], "-", color="#005ec4", linewidth=1.7,
                          label="+3σ")
        axes[i + 1].plot(minus[:, 1], minus[:, 0], "-", color="#b02a2a", linewidth=1.7,
                          label="−3σ")
        axes[i + 1].set_title(f"mode {i}  σ={sigma:.2f}", fontsize=9)
        axes[i + 1].axis("equal")
        axes[i + 1].axhline(0, color="#ccc", lw=0.5)
        axes[i + 1].axvline(0, color="#ccc", lw=0.5)

    axes[1].legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.suptitle("Flagellum shape PCA — mean + top modes at ±3σ", fontsize=10)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
