#!/usr/bin/env bash
# Long pretrain on many_cells with the full unified architecture + DETR-style deep supervision.
# Cells are all the same type → no GroupedDecoder (n_groups=1, vanilla decoder).
# Tests whether deep supervision smooths the L_pres / L_where instabilities seen in flagella.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

RUN_DIR=runs/unified_many_cells_v2
STEPS=${STEPS:-100000}
BATCH=${BATCH:-4}
LR=${LR:-2e-4}
N_MAX=${N_MAX:-48}
GLIMPSE=${GLIMPSE:-16}      # cells are round, 16-px patch is plenty
LAYERS=${LAYERS:-3}         # ≥3 so deep supervision has ≥2 intermediate layers
LAMBDA_AUX=${LAMBDA_AUX:-0.5}

echo "=== unified many_cells pretrain (deep supervision, no groups) ==="
echo "steps=$STEPS  batch=$BATCH  lr=$LR  n_max=$N_MAX"
echo "glimpse=$GLIMPSE  layers=$LAYERS  lambda_aux=$LAMBDA_AUX"
echo "run_dir=$RUN_DIR"
date

python3 -m sim2real.scripts.pretrain \
    --sim many_cells \
    --steps "$STEPS" \
    --batch "$BATCH" \
    --lr "$LR" \
    --n-max "$N_MAX" \
    --teacher-force-zpres \
    --t-curriculum --t-start 3 --t-curriculum-steps 25000 \
    --lambda-aux "$LAMBDA_AUX" \
    --glimpse-size "$GLIMPSE" \
    --n-transformer-layers "$LAYERS" \
    --lambda-recon 2.0 \
    --lambda-where 1.0 \
    --lambda-pres 1.0 \
    --lambda-mask 0.0 \
    --lambda-mask-glimpse 10.0 \
    --log-every 250 \
    --ckpt-every 10000 \
    --run-dir "$RUN_DIR"

LAST_CKPT=$(ls -1 "$RUN_DIR"/ckpts/ | sort -V | tail -n 1)
echo
echo "=== Visualizing $LAST_CKPT ==="
python3 -m sim2real.scripts.viz_ckpt \
    --ckpt "$RUN_DIR/ckpts/$LAST_CKPT" \
    --sim many_cells --n-max "$N_MAX" \
    --glimpse-size "$GLIMPSE" --n-transformer-layers "$LAYERS" \
    --out "$RUN_DIR/final_viz.png" || echo "viz failed"

echo "=== Eval metrics ==="
python3 -m sim2real.scripts.eval_ckpt \
    --ckpt "$RUN_DIR/ckpts/$LAST_CKPT" \
    --sim many_cells --n-max "$N_MAX" \
    --glimpse-size "$GLIMPSE" --n-transformer-layers "$LAYERS" \
    --out "$RUN_DIR/eval_metrics.json" || echo "eval failed"

date
echo "DONE: $RUN_DIR"
