"""Side-by-side: real-frame + label overlay | sim-frame + GT skeleton overlay.
Lets you eyeball whether the sim geometry matches real."""
from __future__ import annotations

import json
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image

from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip

ANN = Path("/home/frans/sim2real_tracking/annotations/flagella_v0")
OUT = Path("/home/frans/sim2real_tracking/runs/labels_vs_sim.png")

N_REAL = 16
N_SIM = 16
H = W = 200


def load_real_labeled(k: int):
    m = json.load(open(ANN / "manifest.json"))
    entry = m[k]
    src = entry["source"]
    ct, cb = int(entry["crop_top"]), int(entry["crop_bot"])
    img = tifffile.imread(src) if src.endswith(".tif") else np.array(Image.open(src))
    if img.ndim == 3:
        img = img.mean(-1)
    # Frame in native (source) crop; the mask is in the 512-canonical canvas.
    frame = img[ct:cb].astype(np.float32)
    frame /= 255.0
    canv_mask_flag = np.array(Image.open(ANN / "masks" / f"mask_{k:03d}.png"))
    if canv_mask_flag.ndim == 3:
        canv_mask_flag = canv_mask_flag[..., 0]
    canv_mask_pip_path = ANN / "masks_pipette" / f"mask_{k:03d}.png"
    canv_mask_pip = (np.array(Image.open(canv_mask_pip_path))
                     if canv_mask_pip_path.exists() else np.zeros((512, 512), np.uint8))
    if canv_mask_pip.ndim == 3:
        canv_mask_pip = canv_mask_pip[..., 0]
    canv_mask_body_path = ANN / "masks_body" / f"mask_{k:03d}.png"
    canv_mask_body = (np.array(Image.open(canv_mask_body_path))
                      if canv_mask_body_path.exists() else np.zeros((512, 512), np.uint8))
    if canv_mask_body.ndim == 3:
        canv_mask_body = canv_mask_body[..., 0]

    # Canvas → native transform: undo scale + pad from manifest
    scale = entry["scale"]
    pt = entry["pad_top"]; pl = entry["pad_left"]
    oh, ow = entry["orig_h"], entry["orig_w"]

    def canv_to_native(mask):
        # crop pad and resize back
        m = mask.astype(np.float32)
        m_native = m[pt : pt + int(round(oh * scale)),
                     pl : pl + int(round(ow * scale))]
        # resize by 1/scale to native
        from PIL import Image as PI
        pil = PI.fromarray(m_native)
        pil = pil.resize((ow, oh), PI.NEAREST)
        return np.array(pil)

    mflag = canv_to_native(canv_mask_flag) > 127
    mpip = canv_to_native(canv_mask_pip) > 127
    mbody = canv_to_native(canv_mask_body) > 127
    return frame, mflag, mpip, mbody, entry["out_name"]


def rgb_gray(f):
    lo, hi = np.percentile(f, 1), np.percentile(f, 99)
    return np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1)


def overlay(base_gray, mask_dict):
    """base_gray: (H, W) in [0,1]; mask_dict: {label: (mask, rgb)}"""
    rgb = np.stack([base_gray] * 3, -1)
    for _label, (m, color) in mask_dict.items():
        if m is None or m.sum() == 0:
            continue
        # Ensure mask is same shape as base
        H, W = base_gray.shape
        if m.shape != (H, W):
            from PIL import Image as PI
            m = np.array(PI.fromarray(m.astype(np.uint8) * 255).resize((W, H))) > 127
        for c in range(3):
            rgb[..., c] = np.where(m, 0.55 * rgb[..., c] + 0.45 * color[c], rgb[..., c])
    return rgb


def main():
    # Pick real samples that span DIFFERENT sequences (not consecutive frames
    # from one). We take one frame per sequence, up to N_REAL sequences.
    m_all = json.load(open(ANN / "manifest.json"))
    seen_seqs = {}
    for entry in m_all:
        if entry["sequence"] not in seen_seqs:
            seen_seqs[entry["sequence"]] = entry["idx"]
        if len(seen_seqs) >= N_REAL: break
    real_indices = list(seen_seqs.values())
    real_rows = []
    for i in real_indices:
        try:
            frame, mflag, mpip, mbody, name = load_real_labeled(i)
        except Exception as e:
            print(f"skip real {i}: {e}"); continue
        real_rows.append((frame, mflag, mpip, mbody, name))
        print(f"real {i}: {name}  frame {frame.shape}  "
              f"flag={mflag.sum()} pip={mpip.sum()} body={mbody.sum()}")

    # Sim samples
    cfg = DiverseSimConfig(T=16, H=H, W=W)
    print("compiling sim...")
    sim_rows = []
    for i in range(N_SIM):
        out = sample_clip(jax.random.key(2000 + i), cfg)
        raw = np.asarray(out["clip_raw"])
        t = raw.shape[0] // 2
        # GT overlay: flagellum skeleton lines + cell circles + pipette line
        cells = out["cells"]
        flag = out["flagella"]
        pip = out["pipette"]
        # We'll rasterise gt as three masks the same size as sim frame
        gt_flag = np.zeros((H, W), bool)
        gt_body = np.zeros((H, W), bool)
        gt_pip = np.zeros((H, W), bool)
        # Cell body as filled disks
        yy, xx = np.mgrid[:H, :W]
        for cy, cx, r, alive in zip(np.asarray(cells["centers"])[:, 0],
                                     np.asarray(cells["centers"])[:, 1],
                                     np.asarray(cells["radii"]),
                                     np.asarray(cells["alive"])):
            if alive:
                gt_body |= ((yy - cy) ** 2 + (xx - cx) ** 2) < r * r
        # Flagellum skeletons — draw each slot in its own distinct color
        # so paired flagella don't visually merge into a single GT overlay.
        curves = np.asarray(out["curves"])[t]   # (N_flag, K, 2)
        from scipy.ndimage import binary_dilation
        gt_flag_by_slot = []   # list of (mask, rgb) per alive slot
        slot_colors = [
            (0.20, 1.00, 0.30),   # slot 0: green
            (0.15, 0.85, 1.00),   # slot 1: cyan
            (1.00, 0.55, 0.20),   # slot 2: orange
            (1.00, 0.20, 0.90),   # slot 3: magenta
        ]
        for k in range(curves.shape[0]):
            if not bool(flag["alive"][k]):
                continue
            pts = np.round(curves[k]).astype(int)
            m = np.zeros((H, W), bool)
            for j in range(pts.shape[0] - 1):
                y0, x0 = pts[j]; y1, x1 = pts[j + 1]
                steps = max(abs(y1 - y0), abs(x1 - x0)) + 1
                ys = np.linspace(y0, y1, steps).astype(int)
                xs = np.linspace(x0, x1, steps).astype(int)
                mm = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
                m[ys[mm], xs[mm]] = True
            m = binary_dilation(m, iterations=1)
            gt_flag_by_slot.append((m, slot_colors[k]))
        # Keep gt_flag as union for backward-compat (used by overlay dict)
        gt_flag = np.any([m for m, _ in gt_flag_by_slot], axis=0) if gt_flag_by_slot \
                  else gt_flag
        # Pipette line
        if bool(pip["present"]):
            p0 = np.asarray(pip["base"])
            p1 = np.asarray(pip["tip"])
            steps = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
            ys = np.linspace(p0[0], p1[0], steps).astype(int)
            xs = np.linspace(p0[1], p1[1], steps).astype(int)
            m = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
            gt_pip[ys[m], xs[m]] = True
            gt_pip = binary_dilation(gt_pip, iterations=1)
        sim_rows.append((raw[t], gt_flag_by_slot, gt_pip, gt_body,
                          f"sim seed={2000 + i}"))

    n = max(len(real_rows), len(sim_rows))
    fig, axes = plt.subplots(n, 4, figsize=(11, 2.4 * n), squeeze=False)
    for r in range(n):
        for c in range(4):
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
        if r < len(real_rows):
            frame, mflag, mpip, mbody, name = real_rows[r]
            g = rgb_gray(frame)
            axes[r][0].imshow(g, cmap="gray")
            axes[r][1].imshow(overlay(g, {"body": (mbody, (0.35, 0.85, 1.0)),
                                            "pip": (mpip, (1.0, 0.75, 0.15)),
                                            "flag": (mflag, (0.3, 1.0, 0.3))}))
            axes[r][0].set_ylabel(f"REAL: {name}", fontsize=7.5)
        else:
            for c in (0, 1):
                axes[r][c].axis("off")
        if r < len(sim_rows):
            frame, gt_by_slot, mpip, mbody, name = sim_rows[r]
            g = rgb_gray(frame)
            axes[r][2].imshow(g, cmap="gray")
            # Build overlay with each flag slot in its OWN color so pairs
            # are visually distinguishable rather than merged into one blob.
            layers = {"body": (mbody, (0.35, 0.85, 1.0)),
                       "pip": (mpip, (1.0, 0.75, 0.15))}
            for si, (m, col) in enumerate(gt_by_slot):
                layers[f"flag_{si}"] = (m, col)
            axes[r][3].imshow(overlay(g, layers))
        else:
            for c in (2, 3):
                axes[r][c].axis("off")
        if r == 0:
            axes[r][0].set_title("real frame", fontsize=9)
            axes[r][1].set_title("real  +  labels", fontsize=9)
            axes[r][2].set_title("sim frame", fontsize=9)
            axes[r][3].set_title("sim  +  GT", fontsize=9)

    fig.suptitle("Real (labels overlaid) vs sim (GT overlaid)   "
                 "— green=flag, yellow=pipette, cyan=body",
                 fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
