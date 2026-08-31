"""Illustration for the energy-score explainer.

Three predictive point clouds versus a fixed target y:
  (A) Point-mass at the wrong location — low spread, bad accuracy.
  (B) Delta at the truth — perfect but overconfident (score = 0).
  (C) Diverse cloud around the truth — modest ‖X−y‖ paid off by large ½‖X−X'‖.
The energy score is minimized by a calibrated (not collapsed) predictive.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)
y = np.array([0.0, 0.0])

def sample(mean, cov, n=200):
    return rng.multivariate_normal(mean, cov, size=n)

samples = [
    dict(label="(A) confident + wrong",
         pts=sample([1.6, 0.4], 0.02 * np.eye(2)),
         color="#b02a2a"),
    dict(label="(B) collapsed at truth",
         pts=sample([0.0, 0.0], 0.005 * np.eye(2)),
         color="#178d3f"),
    dict(label="(C) diverse near truth",
         pts=sample([0.05, -0.05], 0.35 * np.eye(2)),
         color="#005ec4"),
]

def energy_score(pts, y):
    """E‖X − y‖ − ½ E‖X − X'‖ (Monte-Carlo)."""
    d_y = np.linalg.norm(pts - y, axis=1).mean()
    diff = pts[:, None, :] - pts[None, :, :]
    d_xx = np.linalg.norm(diff, axis=-1).mean()
    return d_y - 0.5 * d_xx, d_y, 0.5 * d_xx

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), sharex=True, sharey=True)
for ax, s in zip(axes, samples):
    ax.scatter(s["pts"][:, 0], s["pts"][:, 1], s=10, alpha=0.35, color=s["color"],
               edgecolors="none")
    ax.scatter(*y, s=120, marker="*", color="#000", zorder=3, label="target y")
    es, a, b = energy_score(s["pts"], y)
    ax.set_title(
        f"{s['label']}\n"
        f"E‖X−y‖ = {a:.2f}   ½E‖X−X'‖ = {b:.2f}\n"
        f"energy score = {es:.2f}",
        fontsize=9.5)
    ax.set_xlim(-1.6, 2.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#dcdce0")

fig.suptitle("Energy score  E‖X−y‖ − ½ E‖X−X'‖   —   lower is better",
             fontsize=11)
fig.tight_layout()
fig.savefig("/home/frans/sim2real_tracking/docs/assets/energy_score.png", dpi=150,
            bbox_inches="tight")
print("wrote energy_score.png")
