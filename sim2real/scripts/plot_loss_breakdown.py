"""Parse a pretrain run.log and plot every loss term — raw and weighted contribution.

Usage:
    PYTHONPATH=. python3 -m sim2real.scripts.plot_loss_breakdown \
        --log runs/pretrain_many_cells_long/run.log \
        --out runs/pretrain_many_cells_long/loss_breakdown.png \
        --lambda-recon 2.0 --lambda-where 1.0 --lambda-pres 1.0 --lambda-mask 3.0
"""

from __future__ import annotations

import argparse
import re

import numpy as np


LINE_RE = re.compile(
    r"step\s+(?P<step>\d+)\s+loss\s+(?P<loss>[\d.eE+-]+)\s+"
    r"recon\s+(?P<recon>[\d.eE+-]+)\s+"
    r"where\s+(?P<where>[\d.eE+-]+)\s+"
    r"pres\s+(?P<pres>[\d.eE+-]+)\s+"
    r"mask\s+(?P<mask>[\d.eE+-]+)\s+"
    r"gnorm\s+(?P<gnorm>[\d.eE+-]+)"
)


def parse(path):
    rows = []
    with open(path) as f:
        for ln in f:
            m = LINE_RE.search(ln)
            if not m:
                continue
            rows.append({k: float(v) for k, v in m.groupdict().items()})
    if not rows:
        raise RuntimeError(f"no step lines parsed from {path}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lambda-recon", type=float, default=2.0)
    ap.add_argument("--lambda-where", type=float, default=1.0)
    ap.add_argument("--lambda-pres", type=float, default=1.0)
    ap.add_argument("--lambda-mask", type=float, default=3.0)
    args = ap.parse_args()

    rows = parse(args.log)
    step = np.array([r["step"] for r in rows])
    L_total = np.array([r["loss"] for r in rows])
    L_recon = np.array([r["recon"] for r in rows])
    L_where = np.array([r["where"] for r in rows])
    L_pres = np.array([r["pres"] for r in rows])
    L_mask = np.array([r["mask"] for r in rows])
    gnorm = np.array([r["gnorm"] for r in rows])

    # Weighted contributions to the total.
    c_recon = args.lambda_recon * L_recon
    c_where = args.lambda_where * L_where
    c_pres = args.lambda_pres * L_pres
    c_mask = args.lambda_mask * L_mask
    sum_contrib = c_recon + c_where + c_pres + c_mask

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (0,0) total loss + recovered sum-of-contributions (sanity)
    ax = axes[0, 0]
    ax.plot(step, L_total, label="total (logged)", lw=2)
    ax.plot(step, sum_contrib, label="Σ λ·L (reconstructed)", lw=1, ls="--")
    ax.set_title("Total loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # (0,1) raw per-term values (no λ applied) — apples-to-apples for inspecting per-term progress
    ax = axes[0, 1]
    ax.plot(step, L_recon, label="L_recon")
    ax.plot(step, L_where, label="L_where")
    ax.plot(step, L_pres, label="L_pres")
    ax.plot(step, L_mask, label="L_mask")
    ax.set_title("Raw per-term loss (no λ)")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, which="both")

    # (1,0) weighted contributions on linear scale — shows what's actually driving the total
    ax = axes[1, 0]
    ax.stackplot(
        step,
        c_recon, c_where, c_pres, c_mask,
        labels=[
            f"λ_recon·L_recon ({args.lambda_recon})",
            f"λ_where·L_where ({args.lambda_where})",
            f"λ_pres·L_pres ({args.lambda_pres})",
            f"λ_mask·L_mask ({args.lambda_mask})",
        ],
        alpha=0.85,
    )
    ax.plot(step, L_total, color="k", lw=1, label="total (logged)")
    ax.set_title("Weighted contributions (stacked)")
    ax.set_xlabel("step")
    ax.set_ylabel("contribution")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (1,1) gradient norm
    ax = axes[1, 1]
    ax.plot(step, gnorm, color="tab:red")
    ax.set_title("Gradient global-norm (post-clip target 1.0)")
    ax.set_xlabel("step")
    ax.set_ylabel("‖g‖")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"sim2real_tracking pretrain breakdown — {len(rows)} log points",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
