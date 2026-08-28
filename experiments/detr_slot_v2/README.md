# DETR-Slot v2 + CP-SAT Verifier — Chlamydomonas flagellum tracking

See [`explainer.html`](./explainer.html) for the full walk-through with visuals.

## What's new vs v1

- **Cell body class in sim + model**: adds `CellHead` (center + radius + amp, each with mean+σ) and cell rendering in `sim_flagella.py`. Simulator now samples 1-2 cells per scene, dark disks.
- **Boundary-attached flagella (approach #1)**: sim flagella no longer float freely — attachment lies on a cell's membrane at a sampled angle. Model learns this pattern from the training distribution.
- **CP-SAT verifier**: takes model-sampled candidate curves, renders each, scores by temporal-energy alignment, and picks a non-overlapping subset with the attachment-on-cell constraint enforced as an IP constraint.

## Numbers (v2 5k checkpoint, 7 min of training)

Sample-coverage-recall on 59 real annotated frames (104 GT flagella):

|                 | k=2 (8 canonical px) | k=3 (12 px) | k=5 (20 px) |
|:---------------:|:-------------------:|:-----------:|:-----------:|
| v1 (20k steps)  | 12.5%               | 51.0%       | 99.0%       |
| **v2 (5k steps)** | **50.0%**         | **77.9%**   | **97.1%**   |

*Coverage measures whether **any** of 100 sampled candidates per slot lands within
k canonical-widths of GT. This is what the verifier gets to pick from.*

**CP-SAT verifier** (single-frame selection from ~160 candidates):

| Metric | Value |
|--------|-------|
| Precision | 0.46 |
| Recall    | 0.44 |
| F1        | **0.45** |

Sequences where cell + flagella conform to sim distribution (CC124 family) get
P/R ≈ 0.6/0.7; wetransfer and paradeb sequences degrade because their morphology
falls outside the current sim's cell size / SNR ranges.

## Files

- `explainer.html` — walk-through with visuals
- `detr_slot_v2_ckpt.pkl` — trained model checkpoint (2.4 MB, 500K params)
- `loss_log.json` — per-step training loss history
- `eval_20k_k{2,3,5}.json` — coverage-recall on all 59 annotations
- `verify_5k_energy.json` — verifier P/R/F1 per frame + per sequence
- `viz_raw/` — 8 model predictions on raw source frames
- `viz_sim/` — 12 in-distribution predictions
- `viz_verify/` — 12 verifier selections vs GT

## Reproduce

```bash
pip install -e .
python scripts/harvest_bg_patches.py                       # ~30 s
python scripts/calibrate_flagellum.py                       # ~5 s
python -m sim2real.train_v2.train --steps 5000 --out-dir runs/detr_slot/v2   # ~7 min on 5090
python -m sim2real.eval_v2.run_eval \
    --ckpt experiments/detr_slot_v2/detr_slot_v2_ckpt.pkl --coverage-k 3
python -m sim2real.verifier.run_verify \
    --ckpt experiments/detr_slot_v2/detr_slot_v2_ckpt.pkl \
    --scoring energy --max-flagella 2 --max-cells 1 \
    --birth-flag 1.5 --birth-cell 2.0 --attach-slack 15
```

## What's next (v3, not yet built)

- **Approach #3**: structural head that predicts flagellum attachment as
  `(cell_slot_index, angle_offset)` — attention over cell slots links flagella to
  their origin cell.
- **Multi-frame verifier**: score candidates against the full clip via beat
  rollout, not just the temporal-min projection.
- **Temporal linking across clips**: CP-SAT flow-conservation for track identity
  and length consensus (needs 3-4 more clips per sequence in the sampled set).
- **Widen sim ranges** to cover wetransfer/paradeb morphology (cell radius,
  motion-blur envelope, higher-native-res sequences).
