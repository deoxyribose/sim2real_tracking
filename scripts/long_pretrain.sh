#!/usr/bin/env bash
# Long pretrain run on many_cells, with bumped mask/pres weights to address the diffuse-mask /
# stuck-pres issues seen in the 500-step verification.
#
# After training, render a viz against the final checkpoint and report eval metrics.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

RUN_DIR=runs/pretrain_many_cells_long
STEPS=${STEPS:-20000}
BATCH=${BATCH:-4}
LR=${LR:-2e-4}

echo "=== Starting long pretrain ==="
echo "steps=$STEPS  batch=$BATCH  lr=$LR  run_dir=$RUN_DIR"
date

python3 -m sim2real.scripts.pretrain \
    --sim many_cells \
    --steps "$STEPS" \
    --batch "$BATCH" \
    --lr "$LR" \
    --n-max 48 \
    --lambda-recon 2.0 \
    --lambda-where 1.0 \
    --lambda-pres 1.0 \
    --lambda-mask 3.0 \
    --log-every 100 \
    --ckpt-every 2000 \
    --run-dir "$RUN_DIR"

LAST_CKPT=$(ls -1 "$RUN_DIR"/ckpts/ | sort -V | tail -n 1)
echo
echo "=== Visualizing $LAST_CKPT ==="
python3 -m sim2real.scripts.viz_ckpt \
    --ckpt "$RUN_DIR/ckpts/$LAST_CKPT" \
    --sim many_cells --n-max 48 \
    --out "$RUN_DIR/final_viz.png" || echo "viz failed"

echo
echo "=== Eval metrics ==="
python3 -m sim2real.scripts.eval_ckpt \
    --ckpt "$RUN_DIR/ckpts/$LAST_CKPT" \
    --sim many_cells --n-max 48 \
    --out "$RUN_DIR/eval_metrics.json" || echo "eval failed"

date
echo "DONE: $RUN_DIR"
