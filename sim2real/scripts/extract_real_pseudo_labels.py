"""Run the trained model + DIR on a set of real sequences and dump the
DIR-selected (skeleton, width, amp) per anchor frame as a compact npz for
use as pseudo-labels in a self-training round."""
from __future__ import annotations

import argparse
import glob
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tifffile
from PIL import Image

from sim2real.data.canonicalize import CanonicalConfig, canonicalize_clip
from sim2real.dir.build_problem import BuildConfig, Hypothesis, build_problem
from sim2real.dir.run_dir import load_model_ckpt, load_pca_scaled
from sim2real.dir.solve import SolveConfig, solve_problem
from sim2real.model.unet_energy import (
    UNetEnergy, decode_curves, sample_batched_noise, unpack_pred,
)


def load_frame(path: str) -> np.ndarray:
    if path.endswith(".tif"):
        img = tifffile.imread(path)
    else:
        img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img.mean(-1)
    return img.astype(np.float32)


def process_sequence(ckpt_path: str, pca_path: str, seq_dir: Path, ext: str,
                     crop_bottom: int, src_width_px: float,
                     n_frames: int, start: int, n_draws: int,
                     score_thresh: float, top_k: int) -> dict:
    params, cfg_u = load_model_ckpt(ckpt_path)
    pca_mean, pca_basis = load_pca_scaled(pca_path)
    model = UNetEnergy(cfg=cfg_u)

    files = sorted(glob.glob(str(seq_dir / f"*{ext}")))[start : start + n_frames]
    if not files:
        raise FileNotFoundError(f"no {ext} in {seq_dir}")
    stack = np.stack([load_frame(f)[:crop_bottom] for f in files]) / 255.0

    canon = canonicalize_clip(stack, CanonicalConfig(src_width_px=src_width_px,
                                                       bg_median_window=15))
    residual = canon["clip"]
    if residual.shape[1] != cfg_u.H or residual.shape[2] != cfg_u.W:
        ph = cfg_u.H - residual.shape[1]; pw = cfg_u.W - residual.shape[2]
        if ph < 0 or pw < 0:
            return dict(seq=str(seq_dir), skipped=True)
        residual = np.pad(residual, ((0, 0), (0, ph), (0, pw)), constant_values=0.0)

    stride = max(1, cfg_u.T // 4)
    key = jax.random.key(0)
    per_anchor_res = []
    per_anchor_hypos: list[list[Hypothesis]] = []
    anchors: list[int] = []
    for t0 in range(0, residual.shape[0] - cfg_u.T + 1, stride):
        anchor = t0 + cfg_u.T // 2
        anchors.append(anchor)
        per_anchor_res.append(residual[anchor])
        win = residual[t0 : t0 + cfg_u.T]
        video = jnp.asarray(win)[None]
        cands = []
        for _ in range(n_draws):
            key, k = jax.random.split(key)
            noise = sample_batched_noise(k, 1, cfg_u)
            pred = model.apply(params, video, noise, train=False)
            curves = np.asarray(decode_curves(pred, cfg_u, pca_mean, pca_basis))[0]
            f = unpack_pred(pred)
            w = np.asarray(f["width"][0]).ravel()
            a = np.asarray(f["amp"][0]).ravel()
            s = np.asarray(jax.nn.sigmoid(f["score"][0])).ravel()
            flat_c = curves.reshape(-1, curves.shape[-2], 2)
            keep = np.where(s >= score_thresh)[0]
            if len(keep) > top_k:
                keep = keep[np.argsort(-s[keep])[:top_k]]
            for j in keep:
                cands.append(Hypothesis(frame=len(anchors) - 1,
                                         skeleton=flat_c[j].astype(np.float32),
                                         width=float(w[j]), amp=float(a[j]),
                                         score=float(s[j])))
        per_anchor_hypos.append(cands)

    per_anchor_res_arr = np.stack(per_anchor_res, axis=0)
    all_hypos = [h for hs in per_anchor_hypos for h in hs]
    if not all_hypos:
        return dict(seq=str(seq_dir), skipped=True)

    build_cfg = BuildConfig(cost_mode="score_only", pick_cost_base=5.0,
                              score_bonus=100.0, birth_cost=3.0, death_cost=3.0,
                              link_max_dist=25.0)
    problem = build_problem(all_hypos, per_anchor_res_arr, build_cfg)
    sol = solve_problem(problem, SolveConfig(time_limit_s=60.0))

    # Group selected by anchor
    labels_per_anchor: dict[int, list[dict]] = {}
    for si in sol["selected_indices"]:
        h = all_hypos[si]
        labels_per_anchor.setdefault(h.frame, []).append(dict(
            skeleton=h.skeleton.tolist(),
            width=h.width, amp=h.amp, score=h.score,
        ))

    # For training: emit each anchor as (residual_clip, list-of-skeletons).
    # residual_clip is a T-window centred on anchor. Also emit `anchors` list
    # for reproducibility.
    return dict(
        seq=str(seq_dir),
        crop_bottom=int(crop_bottom),
        src_width_px=float(src_width_px),
        residual_shape=[int(x) for x in residual.shape],
        residual=residual.astype(np.float16),   # save space
        anchors=[int(a) for a in anchors],
        labels_per_anchor={int(i): labels_per_anchor.get(i, [])
                            for i in range(len(anchors))},
        n_selected=len(sol["selected_indices"]),
        n_candidates=len(all_hypos),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pca", default="data_cache/flagella_pca.npz")
    ap.add_argument("--manifest", default="annotations/flagella_v0/manifest.json")
    ap.add_argument("--n-sequences", type=int, default=6)
    ap.add_argument("--n-frames-per-seq", type=int, default=64)
    ap.add_argument("--n-draws", type=int, default=4)
    ap.add_argument("--score-thresh", type=float, default=0.03)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m = json.load(open(args.manifest))
    # Take one entry per unique sequence, up to n_sequences
    seen = {}
    for entry in m:
        if entry["sequence"] not in seen:
            seen[entry["sequence"]] = entry
        if len(seen) >= args.n_sequences: break
    entries = list(seen.values())
    print(f"processing {len(entries)} sequences", flush=True)

    all_data = []
    for i, entry in enumerate(entries):
        src = entry["source"]
        seq_dir = Path(src).parent
        ext = "." + src.rsplit(".", 1)[1]
        crop_bot = int(entry["crop_bot"])
        print(f"  [{i}] {seq_dir.name}  ext={ext}  crop_bot={crop_bot}", flush=True)
        try:
            data = process_sequence(
                args.ckpt, args.pca, seq_dir, ext, crop_bot,
                src_width_px=4.0, n_frames=args.n_frames_per_seq,
                start=0, n_draws=args.n_draws, score_thresh=args.score_thresh,
                top_k=args.top_k)
            print(f"     -> n_selected={data.get('n_selected', 0)}/{data.get('n_candidates', 0)}",
                  flush=True)
            all_data.append(data)
        except Exception as e:
            print(f"     -> skipped: {e}", flush=True)

    # Save as a single .npz with per-sequence residuals + labels
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Encoding: put each sequence's residual as its own key, and dump the
    # labels/anchors dict as JSON alongside.
    residuals = {f"res_{i:02d}": np.asarray(d["residual"]) for i, d in enumerate(all_data)}
    meta = [dict(seq=d["seq"], anchors=d["anchors"],
                  labels_per_anchor={str(k): v for k, v in d["labels_per_anchor"].items()},
                  n_selected=d["n_selected"], n_candidates=d["n_candidates"])
             for d in all_data]
    np.savez_compressed(str(out_path), meta=json.dumps(meta), **residuals)
    print(f"wrote {out_path}   sequences={len(all_data)}")


if __name__ == "__main__":
    main()
