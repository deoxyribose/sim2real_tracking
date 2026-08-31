"""Plot recall vs coverage threshold using the raw min_chamfer values
from an eval JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+", required=True,
                    help="pairs of label:path, e.g. v2:runs/energy_v2/eval_full.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    thresholds = np.arange(0, 40, 0.25)
    for spec in args.evals:
        label, path = spec.split(":", 1)
        d = json.load(open(path))
        all_dists = []
        for a in d["per_annotation"]:
            for x in a["pre_dir_min_chamfer"]:
                if x < 1e6:
                    all_dists.append(x)
        all_dists = np.asarray(all_dists)
        recalls = [(all_dists <= t).mean() for t in thresholds]
        ax.plot(thresholds, recalls, "-", linewidth=1.8, label=label)
    ax.axvline(8, color="#888", lw=0.8, linestyle=":")
    ax.axvline(12, color="#888", lw=0.8, linestyle=":")
    ax.text(8, 0.02, "8 px", rotation=90, fontsize=8, color="#888", va="bottom")
    ax.text(12, 0.02, "12 px", rotation=90, fontsize=8, color="#888", va="bottom")
    ax.set_xlabel("coverage threshold (canonical px)")
    ax.set_ylabel("pre-DIR recall")
    ax.set_title("Recall on 104 real flagella vs coverage threshold")
    ax.set_ylim(0, 1.02); ax.set_xlim(0, thresholds[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
