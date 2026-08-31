"""Big view of seed=1000 (first grid sample) — raw + GT overlay explicitly
labeled: green skeleton lines for flagella, yellow for pipette, cyan for
cell body, magenta dots at attachment points."""
from __future__ import annotations

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip

OUT = Path("/home/frans/sim2real_tracking/runs/zoom_seed0.png")


def main():
    cfg = DiverseSimConfig(T=16, H=200, W=200)
    seed = 1007
    out = sample_clip(jax.random.key(seed), cfg)
    raw = np.asarray(out["clip_raw"])
    med = np.asarray(out["clip_median"])
    t = raw.shape[0] // 2
    curves = np.asarray(out["curves"])[t]                 # (N_flag, K, 2)
    flag = out["flagella"]
    cells = out["cells"]
    pip = out["pipette"]
    alive = np.asarray(flag["alive"])
    amps = np.asarray(flag["amp"])
    widths = np.asarray(flag["width"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax in axes:
        ax.imshow(raw[t], cmap="gray")
    axes[0].set_title(f"seed={seed}   raw only", fontsize=11)
    axes[1].set_title("raw + labeled GT   (green=flag, yellow=pipette, cyan=cell)",
                       fontsize=11)

    # cell rim outlines (dashed cyan)
    cx = np.asarray(cells["centers"])
    cr = np.asarray(cells["radii"])
    car = np.asarray(cells["axis_ratio"])
    cor = np.asarray(cells["orientation"])
    for i in range(cfg.n_max_cells):
        if not bool(cells["alive"][i]): continue
        th = np.linspace(0, 2 * np.pi, 200)
        ex = cr[i] * np.cos(th)
        ey = cr[i] * car[i] * np.sin(th)
        rot_y = ex * np.sin(cor[i]) + ey * np.cos(cor[i])
        rot_x = ex * np.cos(cor[i]) - ey * np.sin(cor[i])
        axes[1].plot(cx[i, 1] + rot_x, cx[i, 0] + rot_y, "-", color="#40b0d0",
                     linewidth=1.6, label=("cell rim" if i == 0 else None))
    # flagellum skeletons
    for k in range(cfg.n_max_flagella):
        if not bool(alive[k]): continue
        c = curves[k]
        axes[1].plot(c[:, 1], c[:, 0], "-", color="#33dd33", linewidth=2.5,
                     label=(f"slot {k}"))
        # attachment marker
        axes[1].plot(c[0, 1], c[0, 0], "o", color="#ff33ff", markersize=8)
        axes[1].annotate(f"S{k}: amp={amps[k]:+.2f} w={widths[k]:.2f}",
                          (c[0, 1], c[0, 0]), textcoords="offset points",
                          xytext=(6, 6), color="#ff33ff", fontsize=8)
    # pipette skeleton
    if bool(pip["present"]):
        p0 = np.asarray(pip["base"])
        p1 = np.asarray(pip["tip"])
        axes[1].plot([p0[1], p1[1]], [p0[0], p1[0]], "-", color="#ffcc33",
                     linewidth=2.5, label="pipette")

    axes[1].legend(loc="upper right", fontsize=9)
    for ax in axes:
        ax.set_xlim(0, cfg.W - 1); ax.set_ylim(cfg.H - 1, 0)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"seed 1000 (first grid sample)  —  alive flags = {alive.astype(int).tolist()}",
                  fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
