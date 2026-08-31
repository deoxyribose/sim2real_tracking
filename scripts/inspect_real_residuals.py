"""Median-clean all 16 annotated real sequences and characterize the residual.

Goal: quantify the appearance-distribution the sim needs to cover (plus some).
Outputs
-------
runs/real_residuals/panel.png     — 16-row grid: raw / −global med / −window med
runs/real_residuals/hist.png      — residual-value histograms per sequence, overlaid
runs/real_residuals/stats.json    — per-sequence numeric stats (for calibrating the sim)

Stats we track per sequence
---------------------------
- shape (H, W), n_frames_total, n_frames_loaded
- raw p50 (baseline brightness)
- residual σ via MAD (noise floor)
- residual percentiles at |x| for [50, 90, 99, 99.9]% (signal envelope)
- fraction of pixels above 3σ (how sparse is the residual)
- debris_std: MAD of a very-aggressive rolling median (window ≈ 5) residual
  — everything moving faster than ~cell/pipette drift, including flagella,
  debris, and shot noise. Rough upper bound on distractor-level signal.
- drift_energy: mean |global_median − window_median|, per pixel — how much
  the scene drifts across the loaded window (relevant to whether sim needs
  slow translation of cell/pipette).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image

ANNOT_MANIFEST = Path("/home/frans/sim2real_tracking/annotations/flagella_v0/manifest.json")
OUT_DIR = Path("/home/frans/sim2real_tracking/runs/real_residuals")

N_FRAMES = 200
WINDOW_LONG = 21
WINDOW_SHORT = 5


def load_frame(path: str, crop_top: int, crop_bot: int) -> np.ndarray:
    if path.endswith(".tif"):
        img = tifffile.imread(path)
    else:
        img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img.mean(axis=-1)
    return img[crop_top:crop_bot].astype(np.float32)


def load_stack(dirpath: Path, ext: str, crop_top: int, crop_bot: int,
               n_frames: int) -> np.ndarray:
    files = sorted(glob.glob(str(dirpath / f"*{ext}")))
    if not files:
        raise FileNotFoundError(f"no frames in {dirpath}")
    idxs = np.linspace(0, len(files) - 1, n_frames).astype(int)
    frames = [load_frame(files[i], crop_top, crop_bot) for i in idxs]
    return np.stack(frames), len(files)


def sliding_median(stack: np.ndarray, window: int) -> np.ndarray:
    T = stack.shape[0]
    half = window // 2
    out = np.empty_like(stack)
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        out[t] = np.median(stack[lo:hi], axis=0)
    return out


def mad(x: np.ndarray) -> float:
    m = np.median(x)
    return float(1.4826 * np.median(np.abs(x - m)))


def sigma_from_temporal_diff(stack: np.ndarray) -> float:
    """Robust shot-noise estimate: MAD of (frame_t − frame_{t-1}) / √2.

    For a near-static scene, temporal difference is dominated by independent
    per-frame photon noise, so std(diff) ≈ √2 · σ. Unlike MAD on the median
    residual, this doesn't degenerate to 0 or 1.4826 when noise is smaller
    than the uint8 quantization step (many frames identical → residual = 0,
    but consecutive frames still fluctuate)."""
    d = np.diff(stack, axis=0)
    return mad(d) / np.sqrt(2.0)


def sequences_from_manifest():
    m = json.load(open(ANNOT_MANIFEST))
    seen = {}
    for entry in m:
        seq = entry["sequence"]
        if seq not in seen:
            src = entry["source"]
            ext = "." + src.rsplit(".", 1)[1]
            seen[seq] = dict(
                name=seq,
                dirpath=Path(src).parent,
                ext=ext,
                crop_top=int(entry["crop_top"]),
                crop_bot=int(entry["crop_bot"]),
            )
    # deterministic ordering
    return [seen[k] for k in sorted(seen)]


def rgb_gray(frame: np.ndarray, vmin=None, vmax=None):
    if vmin is None:
        vmin, vmax = float(frame.min()), float(frame.max())
    g = np.clip((frame - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    return g


def rgb_signed(frame: np.ndarray, rng: float):
    return np.clip((frame + rng) / (2 * rng), 0, 1)


def render_panel(rows, out_path: Path):
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(12, 1.6 * n),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1.2]),
                             squeeze=False)
    for r, row in enumerate(rows):
        raw = row["raw_frame"]
        rsub_g = row["res_global_frame"]
        rsub_w = row["res_window_frame"]
        rng = row["render_rng"]

        axes[r][0].imshow(rgb_gray(raw), cmap="gray")
        axes[r][1].imshow(rgb_signed(rsub_g, rng), cmap="seismic")
        axes[r][2].imshow(rgb_signed(rsub_w, rng), cmap="seismic")

        # residual value histogram (log y), on the window-median residual
        vals = row["res_window_all"].ravel()
        axes[r][3].hist(vals, bins=80, range=(-4 * row["sigma"], 4 * row["sigma"]),
                        color="#005ec4", alpha=0.85, density=True)
        axes[r][3].axvline(0, color="#000", linewidth=.5)
        for k in (3, -3):
            axes[r][3].axvline(k * row["sigma"], color="#a86300",
                               linewidth=.5, linestyle="--")
        axes[r][3].set_yscale("log")
        axes[r][3].set_ylim(1e-4, 1)
        axes[r][3].set_xticks([-2 * row["sigma"], 0, 2 * row["sigma"]])
        axes[r][3].tick_params(labelsize=7)
        axes[r][3].set_xlim(-4 * row["sigma"], 4 * row["sigma"])

        axes[r][0].set_ylabel(row["short_name"], fontsize=7.5)

        for c in range(3):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
        if r == 0:
            axes[r][0].set_title("raw", fontsize=9)
            axes[r][1].set_title("− global median", fontsize=9)
            axes[r][2].set_title("− window median", fontsize=9)
            axes[r][3].set_title("residual histogram (log y)", fontsize=9)
        for c in range(4):
            for s in axes[r][c].spines.values():
                s.set_color("#dcdce0")

    fig.suptitle("Residual-distribution audit — 16 annotated real sequences  "
                 f"(N={N_FRAMES} frames each, window={WINDOW_LONG})", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def render_overlay(rows, out_path: Path):
    """Overlay of residual PDFs (window median) — do the sequences cluster or spread?"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for row in rows:
        vals = row["res_window_all"].ravel()
        # scaled by MAD so we can compare shapes across sequences of different exposure
        v = vals / max(row["sigma"], 1e-3)
        axes[0].hist(v, bins=160, range=(-10, 10), density=True,
                     histtype="step", linewidth=1.1, label=row["short_name"])
        axes[1].hist(vals, bins=160, range=(-40, 40), density=True,
                     histtype="step", linewidth=1.1)
    axes[0].axvline(0, color="#000", lw=.4)
    axes[0].set_xlabel("residual / σ_MAD (unitless)")
    axes[0].set_yscale("log"); axes[0].set_ylim(1e-6, 1)
    axes[0].set_title("residual PDF, σ-scaled")
    axes[1].set_xlabel("residual (raw pixel units)")
    axes[1].set_yscale("log"); axes[1].set_ylim(1e-6, 1)
    axes[1].set_title("residual PDF, raw units")
    axes[0].legend(loc="upper right", fontsize=6.5, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    seqs = sequences_from_manifest()
    print(f"processing {len(seqs)} sequences")

    rows = []
    for i, s in enumerate(seqs):
        try:
            stack, n_total = load_stack(s["dirpath"], s["ext"],
                                         s["crop_top"], s["crop_bot"], N_FRAMES)
        except Exception as e:
            print(f"  [{i:2d}] SKIP {s['name']}: {e}")
            continue
        H, W = stack.shape[1:]

        med_global = np.median(stack, axis=0)
        med_win_long = sliding_median(stack, WINDOW_LONG)
        med_win_short = sliding_median(stack, WINDOW_SHORT)

        res_g = stack - med_global
        res_w = stack - med_win_long
        res_debris = stack - med_win_short   # rejects slower-varying signals only

        sigma = sigma_from_temporal_diff(stack)
        # Fallback: if noise is so low it defeats even the diff estimator,
        # clamp to 0.5 (half a uint8 quantum) so downstream ratios stay finite.
        sigma = max(sigma, 0.5)
        raw_p50 = float(np.percentile(stack, 50))
        raw_p5, raw_p95 = np.percentile(stack, [5, 95])
        res_abs_pctiles = np.percentile(np.abs(res_w), [50, 90, 99, 99.9]).tolist()
        frac_above_3sig = float(np.mean(np.abs(res_w) > 3 * sigma))
        # debris estimate: MAD of the short-window residual (removes anything
        # slower than ~cell/pipette drift; keeps flagella + debris + shot noise)
        debris_sigma = mad(res_debris)
        drift_energy = float(np.mean(np.abs(med_global - med_win_long)))
        # signal peak (99.9th percentile relative to noise floor)
        peak_snr = res_abs_pctiles[-1] / max(sigma, 1e-3)

        # short name for row label
        short = s["name"].split("/")[-1][:32]

        # For rendering: middle frame
        t = stack.shape[0] // 2
        rows.append(dict(
            name=s["name"],
            short_name=short,
            raw_frame=stack[t],
            res_global_frame=res_g[t],
            res_window_frame=res_w[t],
            res_window_all=res_w,      # kept for histograms; downstream doesn't consume
            sigma=sigma,
            render_rng=float(np.percentile(np.abs(res_w), 99.5)),
            stats=dict(
                shape=[int(H), int(W)],
                n_frames_total=int(n_total),
                n_frames_loaded=int(stack.shape[0]),
                raw_p5=float(raw_p5), raw_p50=raw_p50, raw_p95=float(raw_p95),
                residual_sigma_mad=float(sigma),
                residual_abs_p50=float(res_abs_pctiles[0]),
                residual_abs_p90=float(res_abs_pctiles[1]),
                residual_abs_p99=float(res_abs_pctiles[2]),
                residual_abs_p999=float(res_abs_pctiles[3]),
                peak_snr=float(peak_snr),
                frac_above_3sigma=frac_above_3sig,
                debris_sigma_mad_windowshort=float(debris_sigma),
                drift_energy_global_vs_window=drift_energy,
            ),
        ))
        print(f"  [{i:2d}] {short:32s} σ={sigma:5.2f}  peak/σ={peak_snr:5.1f}  "
              f"drift={drift_energy:5.2f}")

    if not rows:
        print("no rows to plot"); return

    # Free per-frame stacks that we don't need beyond histograms
    render_panel(rows, OUT_DIR / "panel.png")
    render_overlay(rows, OUT_DIR / "hist.png")

    stats = {r["name"]: r["stats"] for r in rows}
    # aggregate range across sequences — useful for sim calibration
    keys = ["residual_sigma_mad", "residual_abs_p99", "residual_abs_p999",
            "peak_snr", "frac_above_3sigma",
            "debris_sigma_mad_windowshort", "drift_energy_global_vs_window",
            "raw_p5", "raw_p50", "raw_p95"]
    agg = {}
    for k in keys:
        vs = [r["stats"][k] for r in rows]
        agg[k] = dict(min=float(min(vs)), max=float(max(vs)),
                      median=float(np.median(vs)))
    stats["_aggregate"] = agg
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"wrote {OUT_DIR / 'stats.json'}")


if __name__ == "__main__":
    main()
