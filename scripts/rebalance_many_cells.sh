#!/usr/bin/env bash
# Diagnostic: replace full-composite pixel MSE (which plateaus at mean-image) with per-slot
# glimpse-space appearance MSE, foreground-weighted by GT mask.
#
# What changed vs unified_many_cells_v2 (which showed L_recon flat at 0.011 for 100k):
#   --lambda-recon         2.0  -> 0.0   (drop full-composite pixel MSE)
#   --lambda-appear-glimpse 0   -> 10.0  (per-slot glimpse appearance vs GT crop, fg-weighted)
#   --lambda-mask-glimpse  10.0 -> 10.0  (unchanged — mask supervision in glimpse coords)
# Plus the eval hook logs held-out recon/PSNR/SSIM/IoU/silhouette every 500 steps.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=/home/frans/jaxup_venv/bin/python

RUN_DIR=runs/scale_fix_jax010
STEPS=10000
BATCH=4
LR=2e-4
N_MAX=48
GLIMPSE=16
LAYERS=3
LAMBDA_AUX=0.5

mkdir -p "$RUN_DIR"
echo "=== rebalance: drop pixel recon, use glimpse-space appearance MSE ==="
echo "steps=$STEPS batch=$BATCH lr=$LR n_max=$N_MAX"
date

$PY -m sim2real.scripts.pretrain \
    --sim many_cells \
    --steps "$STEPS" \
    --batch "$BATCH" \
    --lr "$LR" \
    --n-max "$N_MAX" \
    --t-curriculum --t-start 3 --t-curriculum-steps 5000 \
    --lambda-aux "$LAMBDA_AUX" \
    --glimpse-size "$GLIMPSE" \
    --n-transformer-layers "$LAYERS" \
    --lambda-recon 2.0 \
    --lambda-where 5.0 \
    --lambda-pres 1.0 \
    --lambda-mask 0.0 \
    --lambda-mask-glimpse 10.0 \
    --lambda-appear-glimpse 10.0 \
    --lambda-slot-contrast 1.0 \
    --slot-contrast-tau 0.1 \
    --matching-mode per_frame \
    --match-once-after 500 \
    --log-every 100 \
    --ckpt-every 200 \
    --eval-every 250 \
    --eval-batch 4 \
    --run-dir "$RUN_DIR"

LAST_CKPT=$(ls -1 "$RUN_DIR"/ckpts/ 2>/dev/null | sort -V | tail -n 1)
if [ -n "$LAST_CKPT" ]; then
    echo "=== Visualizing $LAST_CKPT ==="
    $PY -m sim2real.scripts.viz_ckpt \
        --ckpt "$RUN_DIR/ckpts/$LAST_CKPT" \
        --sim many_cells --n-max "$N_MAX" \
        --glimpse-size "$GLIMPSE" --n-transformer-layers "$LAYERS" \
        --out "$RUN_DIR/final_viz.png" || echo "viz failed"

    echo "=== Eval ==="
    $PY -m sim2real.scripts.eval_ckpt \
        --ckpt "$RUN_DIR/ckpts/$LAST_CKPT" \
        --sim many_cells --n-max "$N_MAX" \
        --glimpse-size "$GLIMPSE" --n-transformer-layers "$LAYERS" \
        --out "$RUN_DIR/eval_metrics.json" || echo "eval failed"
fi
date
echo "DONE: $RUN_DIR"
