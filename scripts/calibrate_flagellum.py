"""Layer-0 calibration: measure flagellum widths, SNR, curvature, length from the 59
painted annotations in `annotations/flagella_v0/`. All measurements in NATIVE pixel
space (i.e. mapped back through the annotation resize+crop transform), so widths
across sequences with different native resolutions are directly comparable.

Emits: `annotations/flagella_v0/calibration.json` — per-sequence stats + global
distribution summary. Used to parameterize:
  - canonicalization width-normalization target σ
  - sim FG-rendering width / SNR / length sampling ranges
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile
from scipy.ndimage import binary_dilation, distance_transform_edt, label as cc_label
from skimage.morphology import skeletonize

ROOT = Path("/home/frans/sim2real_tracking/annotations/flagella_v0")


def canvas_mask_to_cropped(mask_canvas, meta):
    """Undo the annot_sample_frames.py transform back to CROPPED-native space
    (i.e. the image space after banner removal but before resize+pad to canvas).

    Note: the sampler's manifest stores `orig_h`/`orig_w` = the CROPPED height/width
    (post-banner-removal), NOT the raw native image size. So "native" here means
    cropped-native.
    """
    pad_top, pad_left = meta["pad_top"], meta["pad_left"]
    orig_h, orig_w = meta["orig_h"], meta["orig_w"]  # cropped-native dims
    scale = meta["scale"]

    H_pad = int(round(orig_h * scale))
    W_pad = int(round(orig_w * scale))
    unpadded = mask_canvas[pad_top : pad_top + H_pad, pad_left : pad_left + W_pad]
    if unpadded.shape != (orig_h, orig_w):
        im = Image.fromarray((unpadded > 127).astype(np.uint8) * 255)
        im = im.resize((orig_w, orig_h), Image.NEAREST)
        unpadded = np.array(im)
    return unpadded > 127


def load_source_cropped(path, meta):
    """Load source frame and apply the same banner crop as the sampler."""
    if path.endswith(".tif"):
        img = tifffile.imread(path)
    else:
        img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img.mean(axis=-1).astype(np.uint8)
    return img[meta["crop_top"] : meta["crop_bot"], :]


def measure_flagellum(gt_mask: np.ndarray, native_img: np.ndarray, min_len: int = 8) -> list[dict]:
    """Per connected-component measurements: width, length, SNR, curvature.

    - width: 2× median distance-to-background along skeleton pixels.
    - length: skeleton pixel count (approximates arc length in pixels).
    - snr: mean|inside - local_bg| / local_bg_noise_sigma
    - straightness: endpoint_distance / arc_length ∈ [0, 1] (1 = straight, small = curly).
    """
    if not gt_mask.any():
        return []
    # Distance transform on the mask (interior) — half-width at each interior pixel.
    dt = distance_transform_edt(gt_mask.astype(np.uint8))
    # Skeleton
    skel = skeletonize(gt_mask)
    if not skel.any():
        return []

    # Local BG statistics: intensity/noise in a ring around the mask, excluding the mask itself.
    dil = binary_dilation(gt_mask, iterations=6)
    ring = dil & ~gt_mask
    if ring.sum() < 20:
        bg_mean = float(native_img.mean())
        bg_std = float(native_img.std()) + 1e-6
    else:
        bg_mean = float(native_img[ring].mean())
        bg_std = float(native_img[ring].std()) + 1e-6

    # Group by connected component; report per-CC stats.
    labeled, n = cc_label(gt_mask)
    out = []
    for k in range(1, n + 1):
        cc = labeled == k
        skel_k = skel & cc
        if skel_k.sum() < min_len:
            continue
        # Width: 2× median distance-to-background sampled along the skeleton (pixels).
        widths = 2 * dt[skel_k]
        width_med = float(np.median(widths))
        length = int(skel_k.sum())

        # Straightness (endpoint / arc). Find skeleton endpoints: skeleton pixels with exactly 1
        # skeleton neighbor.
        from scipy.ndimage import convolve
        neigh_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neigh_count = convolve(skel_k.astype(np.uint8), neigh_kernel, mode="constant", cval=0)
        endpoints = np.where(skel_k & (neigh_count == 1))
        if len(endpoints[0]) >= 2:
            # take the two most-separated endpoints
            pts = np.stack(endpoints, axis=1)  # (M, 2)
            d2 = ((pts[:, None] - pts[None]) ** 2).sum(-1)
            i, j = np.unravel_index(d2.argmax(), d2.shape)
            straight = float(np.sqrt(d2[i, j]) / max(length, 1))
        else:
            straight = 1.0

        # SNR
        inside_intens = native_img[cc]
        signed_delta = float(bg_mean - inside_intens.mean())  # positive = darker than BG
        snr = abs(signed_delta) / bg_std

        out.append(dict(
            width_px=width_med,
            length_px=length,
            snr=snr,
            polarity=int(np.sign(signed_delta)),   # +1 = darker than BG, -1 = brighter
            straightness=straight,
            bg_mean=bg_mean,
            bg_std=bg_std,
        ))
    return out


def summarize(vals: list[float]) -> dict:
    if not vals:
        return dict(n=0)
    a = np.asarray(vals, dtype=np.float64)
    return dict(
        n=len(a),
        min=float(a.min()), max=float(a.max()),
        p10=float(np.percentile(a, 10)), p50=float(np.percentile(a, 50)),
        p90=float(np.percentile(a, 90)),
        mean=float(a.mean()), std=float(a.std()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "calibration.json"))
    args = ap.parse_args()

    manifest = json.load(open(ROOT / "manifest.json"))
    status = json.load(open(ROOT / "status.json"))
    annotated = [m for m in manifest if status.get(m["out_name"]) == "annotated"]
    print(f"[data] {len(annotated)} annotated frames from {len(set(m['sequence'] for m in annotated))} sequences")

    per_seq_widths = defaultdict(list)
    per_seq_snr = defaultdict(list)
    per_seq_length = defaultdict(list)
    per_seq_straight = defaultdict(list)
    per_seq_polarity = defaultdict(list)
    per_seq_native_hw = defaultdict(list)
    per_frame_records = []
    skipped_frames = 0

    for m in annotated:
        canvas_mask = np.array(Image.open(ROOT / "masks" / m["out_name"].replace("img_", "mask_")))
        try:
            native_mask = canvas_mask_to_cropped(canvas_mask, m)
            native_img = load_source_cropped(m["source"], m)
        except Exception as e:
            print(f"  skip {m['out_name']}: {e}")
            skipped_frames += 1
            continue
        if native_img.shape != native_mask.shape:
            print(f"  skip {m['out_name']}: shape mismatch img={native_img.shape} mask={native_mask.shape}")
            skipped_frames += 1
            continue

        records = measure_flagellum(native_mask, native_img)
        seq = m["sequence"]
        for r in records:
            per_seq_widths[seq].append(r["width_px"])
            per_seq_snr[seq].append(r["snr"])
            per_seq_length[seq].append(r["length_px"])
            per_seq_straight[seq].append(r["straightness"])
            per_seq_polarity[seq].append(r["polarity"])
        per_seq_native_hw[seq].append(native_img.shape)
        per_frame_records.append(dict(frame=m["out_name"], sequence=seq, ccs=records))

    if skipped_frames:
        print(f"  skipped {skipped_frames} frames due to load issues")

    # Per-sequence summary
    per_seq_summary = {}
    for seq in per_seq_widths:
        hws = per_seq_native_hw[seq]
        per_seq_summary[seq] = dict(
            native_h=int(np.median([h for h, w in hws])),
            native_w=int(np.median([w for h, w in hws])),
            width_px=summarize(per_seq_widths[seq]),
            snr=summarize(per_seq_snr[seq]),
            length_px=summarize(per_seq_length[seq]),
            straightness=summarize(per_seq_straight[seq]),
            polarity_pos_frac=float(np.mean([p > 0 for p in per_seq_polarity[seq]])) if per_seq_polarity[seq] else 0.0,
            n_flagella_ccs=len(per_seq_widths[seq]),
        )

    # Global summary
    all_w = [x for v in per_seq_widths.values() for x in v]
    all_s = [x for v in per_seq_snr.values() for x in v]
    all_l = [x for v in per_seq_length.values() for x in v]
    all_st = [x for v in per_seq_straight.values() for x in v]

    global_summary = dict(
        n_frames_measured=len(per_frame_records),
        n_flagella_ccs=len(all_w),
        width_px=summarize(all_w),
        snr=summarize(all_s),
        length_px=summarize(all_l),
        straightness=summarize(all_st),
    )

    print("\n=== GLOBAL ===")
    print(f"  n frames measured: {global_summary['n_frames_measured']}")
    print(f"  n flagellum CCs:   {global_summary['n_flagella_ccs']}")
    for k in ("width_px", "snr", "length_px", "straightness"):
        s = global_summary[k]
        print(f"  {k:14s}  median={s['p50']:.2f}  p10={s['p10']:.2f}  p90={s['p90']:.2f}  min={s['min']:.2f}  max={s['max']:.2f}")

    print("\n=== PER SEQUENCE ===")
    for seq, s in per_seq_summary.items():
        print(f"  {seq}")
        print(f"    native {s['native_h']}x{s['native_w']}, {s['n_flagella_ccs']} CCs")
        print(f"    width_px  med={s['width_px']['p50']:.2f} p10-90=[{s['width_px']['p10']:.2f},{s['width_px']['p90']:.2f}]")
        print(f"    snr       med={s['snr']['p50']:.2f} p10-90=[{s['snr']['p10']:.2f},{s['snr']['p90']:.2f}]")
        print(f"    length_px med={s['length_px']['p50']:.1f} p10-90=[{s['length_px']['p10']:.1f},{s['length_px']['p90']:.1f}]")
        print(f"    straight  med={s['straightness']['p50']:.2f}  polarity+ frac={s['polarity_pos_frac']:.2f}")

    out = dict(
        global_=global_summary,
        per_sequence=per_seq_summary,
        per_frame=per_frame_records,
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
