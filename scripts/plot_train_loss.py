"""Parse training logs and plot loss/accuracy/diversity/score over steps."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LINE_RE = re.compile(
    r"^\[step\s+(\d+)\]\s+t=([\d.]+)s\s+loss=([-\d.nan]+)\s+"
    r"acc=([-\d.nan]+)\s+div=([-\d.nan]+)\s+div/acc=([\d.enan+]+)\s+"
    r"score=([-\d.nan]+)\s+n_gt=([-\d.nan]+)")


def parse_log(path: str) -> dict:
    steps, times, losses, accs, divs, scores = [], [], [], [], [], []
    for line in open(path):
        m = LINE_RE.match(line.strip())
        if not m: continue
        s, t, l, a, d, _, sc, _ = m.groups()
        try:
            steps.append(int(s)); times.append(float(t))
            losses.append(float(l)); accs.append(float(a))
            divs.append(float(d)); scores.append(float(sc))
        except ValueError:
            continue
    return dict(steps=np.asarray(steps), times=np.asarray(times),
                loss=np.asarray(losses), acc=np.asarray(accs),
                div=np.asarray(divs), score=np.asarray(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True,
                    help="label:path pairs, e.g. v3:runs/energy_v3/train.log")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = {}
    for spec in args.logs:
        label, path = spec.split(":", 1)
        runs[label] = parse_log(path)
        print(f"  {label}: {len(runs[label]['steps'])} log lines, "
              f"final loss={runs[label]['loss'][-1]:.2f}   acc={runs[label]['acc'][-1]:.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    colors = plt.cm.tab10.colors

    for ax, key, ylabel, use_log in [
        (axes[0][0], "loss", "total loss", False),
        (axes[0][1], "acc", "coord accuracy (px)", False),
        (axes[1][0], "div", "diversity (px)", False),
        (axes[1][1], "score", "score BCE loss", False),
    ]:
        for i, (label, r) in enumerate(runs.items()):
            # Simple moving average for readability
            y = r[key]
            if len(y) > 10:
                w = max(3, len(y) // 25)
                ymean = np.convolve(y, np.ones(w) / w, mode="valid")
                x = r["steps"][w // 2 : w // 2 + len(ymean)]
                ax.plot(x, ymean, "-", color=colors[i % 10], label=label,
                        linewidth=1.5)
            ax.plot(r["steps"], y, ".", color=colors[i % 10], alpha=0.2, ms=2)
        ax.set_xlabel("step"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if use_log: ax.set_yscale("log")

    axes[0][0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Training curves — smoothed (line) + raw log values (dots)",
                 fontsize=11)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
