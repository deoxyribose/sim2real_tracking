"""Visualize eval-vs-labels: for a sample of annotations, overlay
   - GT flagellum polylines (green)
   - Pre-DIR candidate pool (thin blue)
   - Post-DIR selected (thick orange)
on each canonicalised real frame. Saves one PNG per annotation."""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from sim2real.dir.build_problem import BuildConfig, Hypothesis, build_problem
from sim2real.dir.run_dir import load_model_ckpt, load_pca_scaled
from sim2real.dir.solve import SolveConfig, solve_problem
from sim2real.eval_v2.coverage import (
    canonicalize_real_frame, gt_polyline_to_canonical, load_real_annotations,
)
from sim2real.model.unet_energy import (
    UNetEnergy, decode_curves, sample_batched_noise, unpack_pred,
)
from sim2real.scripts.eval_energy_dir import sample_candidates_for_frame


def rgb_signed(f, rng):
    return np.clip((f + rng) / (2 * rng), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pca", default="data_cache/flagella_pca.npz")
    ap.add_argument("--n-draws", type=int, default=8)
    ap.add_argument("--score-thresh", type=float, default=0.05)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--n-annotations", type=int, default=16,
                    help="viz first N annotations")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params, cfg_u = load_model_ckpt(args.ckpt)
    pca_mean, pca_basis = load_pca_scaled(args.pca)
    model = UNetEnergy(cfg=cfg_u)

    dir_cfg = BuildConfig(cost_mode="score_only", pick_cost_base=3.0,
                            score_bonus=30.0, birth_cost=0.5, death_cost=0.5)
    solve_cfg = SolveConfig(time_limit_s=15.0)

    annots = load_real_annotations()[: args.n_annotations]
    print(f"vizing {len(annots)} annotations", flush=True)

    key = jax.random.key(42)
    rows = []
    for ann in annots:
        try:
            canon, cfg_can = canonicalize_real_frame(ann["meta"],
                                                       ann["src_width_px"],
                                                       T=cfg_u.T)
        except Exception as e:
            print(f"skip {ann['name']}: {e}", flush=True); continue
        clip = canon["clip"]
        if clip.shape[1] != cfg_u.H or clip.shape[2] != cfg_u.W:
            ph = cfg_u.H - clip.shape[1]; pw = cfg_u.W - clip.shape[2]
            if ph < 0 or pw < 0:
                continue
            clip = np.pad(clip, ((0, 0), (0, ph), (0, pw)), constant_values=0.0)
        gt_curves = []
        for pl_native in ann["gt_polylines_native"]:
            gt_canon = gt_polyline_to_canonical(pl_native, ann["meta"], cfg_can,
                                                  canonical_h=cfg_u.H,
                                                  canonical_w=cfg_u.W)
            if len(gt_canon) >= 4:
                gt_curves.append(gt_canon)

        key, kk = jax.random.split(key)
        cand = sample_candidates_for_frame(
            params, model, cfg_u, clip, kk,
            n_draws=args.n_draws, score_thresh=args.score_thresh,
            top_k=args.top_k, pca_mean=pca_mean, pca_basis=pca_basis,
        )
        # Run DIR
        hypos = []
        for i in range(len(cand["scores"])):
            hypos.append(Hypothesis(
                frame=0,
                skeleton=cand["curves"][i].astype(np.float32),
                width=float(cand["widths"][i]),
                amp=float(cand["amps"][i]),
                score=float(cand["scores"][i]),
            ))
        selected = []
        if hypos:
            problem = build_problem(hypos, clip[cfg_u.T // 2 : cfg_u.T // 2 + 1],
                                      dir_cfg)
            sol = solve_problem(problem, solve_cfg)
            selected = sol["selected_indices"]
        rows.append(dict(name=ann["name"], clip=clip, gt=gt_curves,
                          cand=cand, selected=selected))

    n = len(rows)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 6, nrow * 6),
                              squeeze=False)
    for i, row in enumerate(rows):
        r, c = i // ncol, i % ncol
        ax = axes[r][c]
        t = row["clip"].shape[0] // 2
        rng = max(float(np.percentile(np.abs(row["clip"][t]), 99.5)), 0.02)
        ax.imshow(rgb_signed(row["clip"][t], rng), cmap="seismic")

        # Candidate pool: thin light-blue
        for k in range(len(row["cand"]["scores"])):
            cv = row["cand"]["curves"][k]
            ax.plot(cv[:, 1], cv[:, 0], "-", color="#4d90ff", linewidth=0.6,
                    alpha=0.25)
        # Selected: thick orange
        for j in row["selected"][:80]:
            cv = row["cand"]["curves"][j]
            ax.plot(cv[:, 1], cv[:, 0], "-", color="#ff9500", linewidth=1.6,
                    alpha=0.85)
        # GT: thick green
        for gt in row["gt"]:
            ax.plot(gt[:, 1], gt[:, 0], "-", color="#33dd33", linewidth=2.5)

        ax.set_title(f"{row['name']}   n_cand={len(row['cand']['scores'])}   "
                     f"n_sel={len(row['selected'])}   n_gt={len(row['gt'])}",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, cfg_u.W - 1); ax.set_ylim(cfg_u.H - 1, 0)

    for i in range(len(rows), nrow * ncol):
        r, c = i // ncol, i % ncol
        axes[r][c].axis("off")

    fig.suptitle(
        "Eval on real labels — green=GT, orange=DIR-selected, "
        "light-blue=candidate pool", fontsize=11)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
