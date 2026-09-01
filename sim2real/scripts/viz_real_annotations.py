"""Show all annotated real frames with their GT flagellum polylines overlaid.
No model, no eval — just the labels on the raw source frames so we can see
what the model is being asked to recall.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sim2real.eval_v2.coverage import load_real_annotations, load_source_cropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=59, help="up to 59 (all)")
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    annots = load_real_annotations()[: args.n_clips]
    print(f"showing {len(annots)} annotations")
    nrow = (len(annots) + args.ncols - 1) // args.ncols
    fig, axes = plt.subplots(nrow, args.ncols,
                              figsize=(args.ncols * 3.5, nrow * 3.2),
                              squeeze=False)
    for i, ann in enumerate(annots):
        r, c = i // args.ncols, i % args.ncols
        ax = axes[r][c]
        try:
            frame = load_source_cropped(ann["source"], ann["meta"])
        except Exception as e:
            ax.text(0.5, 0.5, f"skip: {e}", ha="center", transform=ax.transAxes)
            ax.axis("off"); continue
        # Grayscale contrast
        lo, hi = np.percentile(frame, [1, 99])
        gray = np.clip((frame - lo) / max(hi - lo, 1e-6), 0, 1)
        ax.imshow(gray, cmap="gray")
        for pl in ann["gt_polylines_native"]:
            ax.plot(pl[:, 1], pl[:, 0], "-", color="#33dd33", linewidth=2.0)
            # attachment endpoint marker
            ax.plot(pl[0, 1], pl[0, 0], "o", color="#ff33ff",
                    markersize=4, markeredgecolor="white", markeredgewidth=0.6)
        ax.set_title(f"{ann['name']}   n_gt={len(ann['gt_polylines_native'])}",
                     fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([])

    for i in range(len(annots), nrow * args.ncols):
        r, c = i // args.ncols, i % args.ncols
        axes[r][c].axis("off")

    fig.suptitle(f"Real annotated frames — green = GT flagellum polylines, "
                 f"magenta = attachment point", fontsize=11)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
