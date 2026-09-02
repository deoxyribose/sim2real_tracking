"""Combine loss/recall curves of the 4 overfit variants into one figure."""
import argparse, pickle, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path):
    """Extract (step, total, coord, score, knot) rows and (step, recall) rows."""
    losses, recalls = [], []
    for line in open(log_path):
        line = line.strip()
        if line.startswith("[") and "L=" in line:
            try:
                step = int(line.split("]")[0].strip("[ "))
                parts = dict(x.split("=") for x in line.split()
                              if "=" in x and x.split("=")[0] in ("L", "c", "s", "k"))
                losses.append((step, float(parts["L"]), float(parts["c"]),
                                float(parts["s"]), float(parts["k"])))
            except Exception: pass
        elif "eval  recall" in line:
            try:
                r = float(line.split("=")[-1].strip())
                if losses: recalls.append((losses[-1][0], r))
            except Exception: pass
    return np.array(losses), np.array(recalls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True,
                    help="pairs: label:logpath  e.g. 'baseline:/tmp/overfit_v8.log' 'A:/tmp/overfit_A.log' ...")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    variants = []
    for spec in args.logs:
        label, path = spec.split(":", 1)
        L, R = parse_log(path)
        variants.append((label, L, R))
        print(f"{label:12s}  {len(L)} loss rows, {len(R)} recall rows")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = ["#4a86e8", "#e69138", "#38761d", "#a64d79"]
    ax = axes[0]
    for i, (label, L, R) in enumerate(variants):
        if len(L) == 0: continue
        ax.plot(L[:, 0], L[:, 1], color=colors[i % len(colors)], label=label)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("total loss")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title("Total loss")

    ax = axes[1]
    for i, (label, L, R) in enumerate(variants):
        if len(L) == 0: continue
        ax.plot(L[:, 0], L[:, 3], color=colors[i % len(colors)], label=label)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("score loss")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title("Score loss (attention pathology)")

    ax = axes[2]
    for i, (label, L, R) in enumerate(variants):
        if len(R) == 0: continue
        ax.plot(R[:, 0], R[:, 1], "-o", color=colors[i % len(colors)], label=label,
                 markersize=4)
    ax.set_xlabel("step"); ax.set_ylabel("rollout recall")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title("Rollout recall on same 8 clips")

    fig.suptitle("Overfit-on-8-clips: baseline vs loss-fix variants (v8 AR from scratch)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
