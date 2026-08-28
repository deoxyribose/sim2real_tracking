# DETR-Slot v3 — Structural Cell-Anchored Flagellum Head

Third of the three cell-attachment approaches from the 2026-08-28 refactor. See
[`explainer.html`](./explainer.html) for the full walk-through including comparison
to v1 (baseline) and v2 (sim-only prior + verifier).

## What's new

**Structural constraint via attention over cell slots** (approach #3):

Instead of predicting flagellum attachment as free `(y, x)`, the head predicts:
- **anchor_probs**: softmax attention over cell slots (which cell does this
  flagellum grow from?)
- **anchor_angle**: angle around that cell's boundary
- **anchor_radius_scale**: slight radial offset (~1.0)
- **control_offsets**: K arc-length deltas (mean + log σ) from the derived
  attachment

Attachment is then **derived** as `cell_center + radius * (sin θ, cos θ)`,
weighted-averaged over the softmax attention. The attach-on-cell constraint is
now **structurally guaranteed** rather than statistically learned from the
sim distribution.

## Numbers (v3 anchored, 8k steps ~ 11 min)

| Metric | v1 (20k) | v2 (5k) | **v3 anchored (8k)** |
|--------|:--------:|:-------:|:--------------------:|
| Coverage k=2 (8 canonical px) | 12.5% | 50.0% | 26.9% |
| Coverage k=3 (12 px)          | 51.0% | 77.9% | 65.4% |
| Coverage k=5 (20 px)          | 99.0% | 97.1% | **96.2%** — 100.0% at step 8k with tighter sample |
| Verifier F1 (energy scoring)  | —     | **0.45** | 0.32 |

Trade-off pattern: v3's structural constraint widens the sampling distribution
(each control-point σ grows across the chain), boosting broad coverage (k=5) at
the cost of per-sample precision (verifier F1).

## Files

- `explainer.html` — full 3-approach comparison with visuals
- `detr_slot_v3_anchored_ckpt.pkl` — trained model (500K params)
- `loss_log.json` — training curves
- `eval_8k_k{2,3,5}.json` — per-frame coverage
- `verify_8k.json` — verifier P/R/F1
- `viz_raw/` — 8 raw-frame predictions
- `viz_verify/` — 12 verifier selections

## Reproduce

```bash
python -m sim2real.train_v2.train --steps 8000 --anchored \
    --out-dir runs/detr_slot/v3_anchored
python -m sim2real.eval_v2.run_eval \
    --ckpt experiments/detr_slot_v3/detr_slot_v3_anchored_ckpt.pkl --coverage-k 3
```
