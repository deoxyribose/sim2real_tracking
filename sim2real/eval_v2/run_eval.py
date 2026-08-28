"""End-to-end evaluation: load ckpt, run model on all 59 real frames, compute
sample-coverage-recall. Emits per-frame + aggregate metrics.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sim2real.data import CANONICAL_H, CANONICAL_W, CANONICAL_TARGET_WIDTH_PX
from sim2real.model_v2 import DETRSlotModel, DETRSlotConfig, sample_flagellum_from_head
from sim2real.eval_v2.coverage import (
    load_real_annotations, canonicalize_real_frame, sample_coverage_recall, gt_polyline_to_canonical,
)


def load_ckpt(path: Path):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--coverage-k", type=float, default=2.0,
                    help="GT covered if any sample within k * canonical_width of GT (Chamfer).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = load_ckpt(Path(args.ckpt))
    model = DETRSlotModel(cfg=ckpt["model_cfg"])
    print(f"[ckpt] step={ckpt['step']}, cfg={ckpt['model_cfg']}")

    ann = load_real_annotations()
    print(f"[data] {len(ann)} real annotated frames")

    @jax.jit
    def forward(params, clip, energy, rng):
        return model.apply(params, clip, energy, rngs={"slots": rng})

    key = jax.random.PRNGKey(0)
    results = []
    total_covered = 0
    total_gt = 0

    for i, entry in enumerate(ann):
        try:
            canon, cfg = canonicalize_real_frame(entry["meta"], entry["src_width_px"], T=args.T)
        except Exception as e:
            print(f"  skip {entry['name']}: {e}")
            continue
        clip = jnp.asarray(canon["clip"])[None]         # (1, T, H, W)
        energy = jnp.asarray(canon["energy"])[None]     # (1, H, W)
        key, sub = jax.random.split(key)
        out = forward(ckpt["params"], clip, energy, sub)

        # Sample candidates
        key, sub2 = jax.random.split(key)
        samples = sample_flagellum_from_head(sub2, out, n_samples=args.n_samples, temperature=args.temperature)
        pts_samples = np.asarray(samples["pts_samples"])[0]  # (S, n_samples, K+1, 2)
        S, N, K1, _ = pts_samples.shape
        pred_all = pts_samples.reshape(S * N, K1, 2)

        # Convert GT native polylines to canonical space (same resample + crop)
        gt_canonical = [gt_polyline_to_canonical(p, entry["meta"], cfg) for p in entry["gt_polylines_native"]]

        metric = sample_coverage_recall(pred_all, gt_canonical, coverage_k=args.coverage_k,
                                         canonical_width_px=CANONICAL_TARGET_WIDTH_PX)
        results.append(dict(
            name=entry["name"], sequence=entry["meta"]["sequence"],
            n_gt=len(gt_canonical), covered=metric["covered"].tolist(),
            min_chamfer=metric["min_chamfer"].tolist(),
            recall=metric["recall"],
        ))
        total_covered += int(metric["covered"].sum())
        total_gt += len(gt_canonical)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(ann)}: running recall = {total_covered}/{total_gt} = {total_covered/max(total_gt,1):.3f}")

    overall = total_covered / max(total_gt, 1)
    print(f"\n[result] overall sample-coverage-recall (k={args.coverage_k}): {overall:.3f} ({total_covered}/{total_gt})")

    # Per-sequence recall
    by_seq: dict = {}
    for r in results:
        s = by_seq.setdefault(r["sequence"], [0, 0])
        s[0] += sum(r["covered"])
        s[1] += r["n_gt"]
    print("\n[per-sequence]")
    for seq, (c, g) in sorted(by_seq.items()):
        print(f"  {seq[-45:]:<45s}  {c}/{g} = {c/max(g,1):.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"overall_recall": overall, "total_covered": total_covered,
                       "total_gt": total_gt, "coverage_k": args.coverage_k,
                       "n_samples_per_slot": args.n_samples, "temperature": args.temperature,
                       "per_frame": results, "per_sequence_recall": {k: v[0]/max(v[1],1) for k,v in by_seq.items()}},
                      f, indent=2)
        print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
