"""Napari-based flagellum annotator.

Load the sampled frames as a stack, add a Labels layer for painting, save masks
as PNG on hotkey and on close.

Usage:
  python scripts/annot_flagella.py --dir /home/frans/sim2real_tracking/annotations/flagella_v0

Controls (all built into napari except where noted):
  - Left drag on the Labels layer with brush tool: paint
  - Alt+drag: erase (Labels layer default)
  - '[' / ']': smaller / larger brush
  - Up/Down arrows: previous / next frame in the stack
  - '1': switch to brush tool
  - '2': switch to fill tool
  - '3': switch to eraser
  - Ctrl+Z / Ctrl+Y: undo / redo
  - 's' (custom): save all masks to disk
  - 'x' (custom): mark current frame as SKIPPED (blank mask, recorded in manifest)
  - 'r' (custom): reset (clear) current frame's mask
Autosave: on window close, all masks are written.
"""
import argparse
import json
import os
from pathlib import Path

import napari
import numpy as np
from PIL import Image


def load_frames(frames_dir: Path):
    files = sorted(frames_dir.glob("img_*.png"))
    imgs = np.stack([np.array(Image.open(f)) for f in files])
    return imgs, [f.name for f in files]


def load_masks(masks_dir: Path, names: list[str], shape: tuple[int, int]) -> np.ndarray:
    stack = np.zeros((len(names), *shape), dtype=np.uint8)
    for i, name in enumerate(names):
        mp = masks_dir / name.replace("img_", "mask_")
        if mp.exists():
            m = np.array(Image.open(mp))
            if m.shape == shape:
                stack[i] = (m > 0).astype(np.uint8)
    return stack


def save_masks(labels: np.ndarray, masks_dir: Path, names: list[str]) -> int:
    masks_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0
    for i, name in enumerate(names):
        out = (labels[i] > 0).astype(np.uint8) * 255
        Image.fromarray(out).save(masks_dir / name.replace("img_", "mask_"))
        if out.any():
            n_saved += 1
    return n_saved


def update_status_manifest(root: Path, names: list[str], labels: np.ndarray, skipped: set[int]):
    """Add annotation status (annotated / skipped / empty) to a status.json alongside manifest."""
    status = {}
    for i, name in enumerate(names):
        if i in skipped:
            status[name] = "skipped"
        elif (labels[i] > 0).any():
            status[name] = "annotated"
        else:
            status[name] = "empty"
    (root / "status.json").write_text(json.dumps(status, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Annotation project dir (contains frames/ and manifest.json)")
    ap.add_argument("--brush-size", type=int, default=4)
    args = ap.parse_args()

    root = Path(args.dir)
    frames_dir = root / "frames"
    masks_dir = root / "masks"

    imgs, names = load_frames(frames_dir)
    print(f"[load] {len(imgs)} frames from {frames_dir}")

    existing = load_masks(masks_dir, names, imgs.shape[1:])
    prev_annotated = int((existing.reshape(len(existing), -1).any(axis=1)).sum())
    print(f"[load] {prev_annotated} frames already have masks (resume)")

    # Load skip list if present
    skipped: set[int] = set()
    status_path = root / "status.json"
    if status_path.exists():
        st = json.loads(status_path.read_text())
        for i, n in enumerate(names):
            if st.get(n) == "skipped":
                skipped.add(i)
        print(f"[load] {len(skipped)} frames previously marked skipped")

    viewer = napari.Viewer(title=f"Flagella annotator — {root.name}")
    img_layer = viewer.add_image(imgs, name="frames", contrast_limits=[0, 255])
    lbl_layer = viewer.add_labels(existing, name="flagellum")
    lbl_layer.brush_size = args.brush_size
    lbl_layer.mode = "paint"
    lbl_layer.selected_label = 1
    # Explicitly restrict painting to the CURRENT 2D slice only (defensive; napari default
    # is 2 but this makes it unambiguous). Without this, some napari builds paint through
    # the T dimension.
    lbl_layer.n_edit_dimensions = 2
    # Select the labels layer so the brush is immediately active.
    viewer.layers.selection.active = lbl_layer

    def _save(*_):
        n = save_masks(lbl_layer.data, masks_dir, names)
        update_status_manifest(root, names, lbl_layer.data, skipped)
        viewer.status = f"saved {n} non-empty masks + status.json"
        print(f"[save] {n} non-empty masks written to {masks_dir}")

    def _skip(*_):
        i = int(viewer.dims.current_step[0])
        skipped.add(i)
        lbl_layer.data[i] = 0
        lbl_layer.refresh()
        viewer.status = f"marked frame {i} ({names[i]}) as SKIPPED"
        print(f"[skip] frame {i} ({names[i]})")

    def _reset(*_):
        i = int(viewer.dims.current_step[0])
        lbl_layer.data[i] = 0
        lbl_layer.refresh()
        skipped.discard(i)
        viewer.status = f"reset frame {i}"

    viewer.bind_key("s", _save, overwrite=True)
    viewer.bind_key("x", _skip, overwrite=True)
    viewer.bind_key("r", _reset, overwrite=True)

    # Auto-save on Qt application quit (fires when the napari window closes).
    from qtpy.QtWidgets import QApplication
    QApplication.instance().aboutToQuit.connect(_save)

    print("[ui] napari open. Keys:  s=save  x=skip current  r=reset current  ↑/↓=nav  [/]=brush size")
    napari.run()


if __name__ == "__main__":
    main()
