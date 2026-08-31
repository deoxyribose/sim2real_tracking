"""Rebuild a clean BG-patch bank from an existing (corrupted) one.

`bg_patches_v0.npz` was harvested before `canonicalize_clip` exposed a validity mask, so
`find_low_energy_centers` preferentially selected center-pad fill (temporal energy exactly
0). Two consequences, both measurable in the saved bank:

  1. 38% of all pixels are constant-zero padding, not background.
  2. σ was estimated over the padded array, collapsing the MAD and pushing 15% of pixels
     onto the ±output_clip_sigma rail. Clipping is lossy — those patches are unrecoverable.

Patches from sequences whose canonical frame needed no padding are untouched by both. This
script keeps exactly those and drops the rest. It is a stopgap: once the raw sequences are
available, re-run `harvest_bg_patches.py` (now valid-mask aware) at --patch-h/--patch-w 256
so no tiling is needed at composite time at all.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def patch_quality(patches: np.ndarray, clip_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (pad_frac, sat_frac) per patch.

    pad_frac: pixels that are exactly 0 in every frame — the center-pad signature.
    sat_frac: pixels sitting on the ±clip_sigma rail, i.e. destroyed by the σ collapse.
    """
    pad = np.all(patches == 0.0, axis=1)
    pad_frac = pad.reshape(len(patches), -1).mean(1)
    sat = np.abs(patches) >= clip_sigma - 1e-3
    sat_frac = sat.reshape(len(patches), -1).mean(1)
    return pad_frac, sat_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data_cache/bg_patches_v0.npz")
    ap.add_argument("--out", default="data_cache/bg_patches_v1.npz")
    ap.add_argument("--clip-sigma", type=float, default=10.0)
    ap.add_argument("--max-pad-frac", type=float, default=0.0)
    ap.add_argument("--max-sat-frac", type=float, default=0.001)
    args = ap.parse_args()

    d = np.load(args.inp, allow_pickle=True)
    patches, seqs = d["patches"], d["sequences"]
    pad_frac, sat_frac = patch_quality(patches, args.clip_sigma)
    keep = (pad_frac <= args.max_pad_frac) & (sat_frac <= args.max_sat_frac)

    print(f"[in ] {args.inp}: {len(patches)} patches, shape {patches.shape[1:]}")
    print(f"       padding pixels {100 * pad_frac.mean():.1f}%   railed pixels {100 * sat_frac.mean():.1f}%")
    print(f"{'sequence':<54s} {'n':>3s} {'kept':>5s} {'pad%':>6s} {'sat%':>6s}")
    for q in sorted(set(seqs.tolist())):
        m = seqs == q
        print(f"{q[-54:]:<54s} {m.sum():3d} {keep[m].sum():5d} "
              f"{100 * pad_frac[m].mean():6.1f} {100 * sat_frac[m].mean():6.1f}")

    kp, ks = patches[keep], seqs[keep]
    pf, sf = patch_quality(kp, args.clip_sigma)
    print(f"\n[out] {len(kp)}/{len(patches)} patches kept from "
          f"{len(set(ks.tolist()))}/{len(set(seqs.tolist()))} sequences")
    print(f"       padding pixels {100 * pf.mean():.3f}%   railed pixels {100 * sf.mean():.3f}%")
    if len(kp) == 0:
        raise SystemExit("no patches survived — refusing to write an empty bank")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, patches=kp.astype(np.float32), sequences=ks,
                        patch_h=d["patch_h"], patch_w=d["patch_w"], clip_len=d["clip_len"])
    print(f"[save] {kp.shape} → {args.out}")


if __name__ == "__main__":
    main()
