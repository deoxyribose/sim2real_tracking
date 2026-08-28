# DETR-Slot v1 — Flagellum tracking with distributional latents

Sim-trained object-centric model on Chlamydomonas flagella. See
[`explainer.html`](./explainer.html) for the walk-through with visuals.

## Numbers (20k training steps, 22 min on RTX 5090)

Sample-coverage-recall on 59 real painted annotations (104 GT flagella), model has
never seen a real frame during training:

| coverage_k | Chamfer threshold | recall |
|:----------:|:-----------------:|:------:|
| 5 | 20 canonical px | **99.0%** (103/104) |
| 3 | 12 canonical px | **51.0%** (53/104) |
| 2 |  8 canonical px | **12.5%** (13/104) |

## Files here

- `explainer.html` — the interactive walk-through
- `detr_slot_v1_ckpt.pkl` — trained model checkpoint (1.9 MB)
- `loss_log.json` — per-step training loss history
- `eval_20k_k{2,3,5}.json` — full per-frame evaluation results
- `viz_raw/` — 8 predictions overlaid on raw source frames
- `viz_sim/` — 12 predictions on in-distribution sim scenes

## Reproduce

```bash
pip install -e .

# Harvest real background patches from the algae videos (one-time, ~30s)
python scripts/harvest_bg_patches.py

# Calibrate flagellum widths / SNR / lengths from the 59 painted annotations
python scripts/calibrate_flagellum.py

# Train — 20k steps, batch=8, ~22 min on 5090
python -m sim2real.train_v2.train --steps 20000 --out-dir runs/detr_slot/v1

# Evaluate on the 59 real frames
python -m sim2real.eval_v2.run_eval \
    --ckpt experiments/detr_slot_v1/detr_slot_v1_ckpt.pkl \
    --n-samples 100 --coverage-k 5.0
```

## The plan document

Original plan for this refactor: this experiment is the first pass. Next up:

1. More sim diversity (wider ranges from `calibration.json`)
2. Cell + pipette classes (currently flagellum-only)
3. Larger encoder (0.48 M is trivially small)
4. SNR curriculum (start high, decay toward measured 0.37 median)
5. Real-replay training once the downstream verifier + CP-SAT loop exists
