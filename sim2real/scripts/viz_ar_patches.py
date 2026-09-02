"""Visualize the 12×12 rotated patches the AR knot generator sees.

For a sim clip, pick the first alive GT flagellum and draw the 24 patch
outlines along its knots (rotated by the tangent). Also render the extracted
patches as a strip so you can see what the model actually gets."""
from __future__ import annotations
import argparse
from pathlib import Path

import jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt

from sim2real.model.unet_ar import UNetARConfig
from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip


def patch_polygon(center_yx, tangent, P):
    """Return the 4 corners (y, x) of a P×P patch centered on `center_yx`,
    rotated so +col direction aligns with tangent."""
    r = (P - 1) / 2.0
    # local corners: (ii, jj) in [-r, r] where ii is row, jj is col (forward)
    corners_local = np.array([[-r, -r], [-r, r], [r, r], [r, -r]])
    c, s = np.cos(tangent), np.sin(tangent)
    # rotate: dy = ii * c + jj * s ; dx = -ii * s + jj * c
    dy = corners_local[:, 0] * c + corners_local[:, 1] * s
    dx = -corners_local[:, 0] * s + corners_local[:, 1] * c
    ys = center_yx[0] + dy
    xs = center_yx[1] + dx
    return np.stack([ys, xs], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-size", type=int, default=12)
    ap.add_argument("--n-knots", type=int, default=24)
    ap.add_argument("--sim-seed", type=int, default=2026)
    ap.add_argument("--try-n", type=int, default=20,
                    help="try up to N clips to find one with a nice flagellum")
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sim_cfg = DiverseSimConfig(T=args.T, H=args.H, W=args.H,
                                 sigma_scale_residual=False)
    key = jax.random.key(args.sim_seed)
    picked = None
    for i in range(args.try_n):
        key, k = jax.random.split(key)
        out = sample_clip(k, sim_cfg)
        alive = np.asarray(out["flagella"]["alive"])
        if alive.sum() >= 1:
            j = int(np.where(alive)[0][0])
            curve = np.asarray(out["curves"])[args.T // 2, j]  # (K_arc, 2)
            picked = dict(idx=i, out=out, j=j, curve=curve)
            if len(curve) >= args.n_knots + 1:
                break
    if picked is None:
        raise SystemExit("no alive flagellum found")

    out = picked["out"]
    curve = picked["curve"]
    img = np.asarray(out["clip_raw"])[args.T // 2].astype(np.float32)

    # Resample the curve to n_knots + 1 points (attachment + 24 knots)
    # (encoder inside the model uses skel[:K] as centers, K = n_knots)
    from sim2real.eval_v2.ordered_metric import resample_polyline
    knots = resample_polyline(curve, args.n_knots + 1)   # (K+1, 2)

    # Compute per-step tangent: tangent of step k = angle from knot k to knot k+1
    # (matches encode_gt_polar_steps: prev_tangent for step k is tangents[k-1],
    # so patch at knot k is rotated by tangents[k-1] = angle of step k-1.
    # For visualization we'll draw the patch at knot k rotated by the incoming
    # tangent tangents[k-1] (with tangent[-1] = 0 for the attachment).)
    steps = np.diff(knots, axis=0)         # (K, 2)
    tangents = np.arctan2(steps[:, 1], steps[:, 0])   # (K,) atan2(dx, dy)
    # But encode_gt_polar_steps uses atan2 differently — check the code.
    # In encode_gt_polar_steps it computes:
    #   d = skel[i+1] - skel[i]
    #   tangent[i] = atan2(d[1], d[0])   # atan2(dx, dy)
    # And patch is rotated so +col = tangent direction, where tangent is measured
    # "clockwise from +y" essentially. So new_pos update uses:
    #   new_pos = pos + step * [sin(tan), cos(tan)]
    # meaning tangent is the "compass" angle from +y axis toward +x.
    # For our polygon: rotate so +col points in direction (sin(tan), cos(tan))
    # relative to (y, x). Our patch_polygon does dy = ii*c + jj*s, dx = -ii*s + jj*c
    # with c=cos(tan), s=sin(tan). For jj=+r: dy = +r*s, dx = +r*c — good, that
    # matches (sin, cos) direction.
    # For the first knot (attachment), tangent = 0 (as in encode_gt_polar_steps).
    K = args.n_knots
    prev_tangents = np.concatenate([[0.0], tangents[:K - 1]])   # (K,)
    centers = knots[:K]                          # (K, 2)

    # --- Panel 1: sim image with patch polygons ---
    fig = plt.figure(figsize=(14, 6.5))
    gs = fig.add_gridspec(2, 8, height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0, :4])
    lo, hi = np.percentile(img, [1, 99])
    gray = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    ax.imshow(gray, cmap="gray")
    # curve
    ax.plot(curve[:, 1], curve[:, 0], "-", color="#33ff44", linewidth=1.2,
            alpha=0.9, label="GT curve")
    ax.plot(knots[:, 1], knots[:, 0], "o", color="#ffcc00", markersize=3,
            label="resampled knots")
    # patch polygons — show a subset for legibility
    show_ks = list(range(0, K, 3))    # every 3rd
    for k in show_ks:
        poly = patch_polygon(centers[k], prev_tangents[k], args.patch_size)
        # matplotlib takes (x, y) tuples
        p = plt.Polygon(poly[:, ::-1], edgecolor="#00e0ff", facecolor="none",
                         linewidth=1.0, alpha=0.75)
        ax.add_patch(p)
        # label knot index at top-left corner
        ax.text(poly[0, 1], poly[0, 0], f"k{k}", fontsize=6, color="#00e0ff")
    ax.set_title(f"sim_{picked['idx']:02d}, flag {picked['j']}: "
                 f"AR knot patches ({args.patch_size}×{args.patch_size} px) "
                 f"every 3rd knot", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8, loc="upper left")

    # Panel 2: zoomed-in view around the flagellum
    ax = fig.add_subplot(gs[0, 4:])
    ax.imshow(gray, cmap="gray")
    ax.plot(curve[:, 1], curve[:, 0], "-", color="#33ff44", linewidth=1.2, alpha=0.9)
    ax.plot(knots[:, 1], knots[:, 0], "o", color="#ffcc00", markersize=3)
    # show ALL patches in zoom
    for k in range(K):
        poly = patch_polygon(centers[k], prev_tangents[k], args.patch_size)
        p = plt.Polygon(poly[:, ::-1], edgecolor="#00e0ff", facecolor="none",
                         linewidth=0.8, alpha=0.6)
        ax.add_patch(p)
    y_lo = max(0, knots[:, 0].min() - 15); y_hi = min(args.H - 1, knots[:, 0].max() + 15)
    x_lo = max(0, knots[:, 1].min() - 15); x_hi = min(args.H - 1, knots[:, 1].max() + 15)
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_hi, y_lo)
    ax.set_title(f"zoomed: all {K} patch outlines", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # Panel 3: extracted patches as a strip
    P = args.patch_size
    n_show = min(K, 8)
    idxs = np.linspace(0, K - 1, n_show).astype(int)
    for kk, k in enumerate(idxs):
        cy, cx = centers[k]
        tan = prev_tangents[k]
        # Sample the RAW image (not features) as a proxy — show what the pixel
        # equivalent of the patch would look like.
        r = (P - 1) / 2.0
        ii = np.arange(P) - r
        jj = np.arange(P) - r
        iim, jjm = np.meshgrid(ii, jj, indexing="ij")
        c, s = np.cos(tan), np.sin(tan)
        dy = iim * c + jjm * s
        dx = -iim * s + jjm * c
        ys = np.clip(cy + dy, 0, args.H - 1)
        xs = np.clip(cx + dx, 0, args.H - 1)
        # bilinear sample
        y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
        y1 = np.clip(y0 + 1, 0, args.H - 1); x1 = np.clip(x0 + 1, 0, args.H - 1)
        wy = ys - y0; wx = xs - x0
        v = (1 - wy) * ((1 - wx) * img[y0, x0] + wx * img[y0, x1]) \
          +      wy * ((1 - wx) * img[y1, x0] + wx * img[y1, x1])
        ax_p = fig.add_subplot(gs[1, kk])
        vlo, vhi = np.percentile(v, [1, 99])
        vshow = np.clip((v - vlo) / max(vhi - vlo, 1e-6), 0, 1)
        ax_p.imshow(vshow, cmap="gray", vmin=0, vmax=1)
        ax_p.set_title(f"k{k}", fontsize=7)
        ax_p.set_xticks([]); ax_p.set_yticks([])

    fig.suptitle(f"AR knot patches on sim: {args.patch_size}×{args.patch_size} px "
                 f"rotated to +col = flagellum tangent. "
                 f"Bottom strip: 8 sampled patches from raw pixels (proxy — "
                 f"real model sees encoder features here).", fontsize=9)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
