#!/usr/bin/env bash
# v4: bg branch + ConvT seg head + lambda_mask bumped to 10.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

RUN_DIR=runs/pretrain_many_cells_2h_v4
STEPS=${STEPS:-100000}
BATCH=${BATCH:-4}
LR=${LR:-2e-4}

echo "=== v4: bg + ConvT seg + lambda_mask=10 ==="
echo "steps=$STEPS  batch=$BATCH  lr=$LR  run_dir=$RUN_DIR"
date

python3 -m sim2real.scripts.pretrain \
    --sim many_cells \
    --steps "$STEPS" \
    --batch "$BATCH" \
    --lr "$LR" \
    --n-max 48 \
    --teacher-force-zwhere \
    --teacher-force-zpres \
    --t-curriculum \
    --t-start 3 \
    --t-curriculum-steps 25000 \
    --lambda-recon 2.0 \
    --lambda-where 1.0 \
    --lambda-pres 1.0 \
    --lambda-mask 10.0 \
    --log-every 250 \
    --ckpt-every 10000 \
    --run-dir "$RUN_DIR"

LAST_CKPT=$(ls -1 "$RUN_DIR"/ckpts/ | sort -V | tail -n 1)
echo
echo "=== Visualizing $LAST_CKPT ==="
python3 -m sim2real.scripts.viz_ckpt \
    --ckpt "$RUN_DIR/ckpts/$LAST_CKPT" \
    --sim many_cells --n-max 48 \
    --out "$RUN_DIR/final_viz.png" || echo "viz failed"

echo
echo "=== Loss breakdown plot ==="
python3 -m sim2real.scripts.plot_loss_breakdown \
    --log "$RUN_DIR/run.log" \
    --out "$RUN_DIR/loss_breakdown.png" \
    --lambda-recon 2.0 --lambda-where 1.0 --lambda-pres 1.0 --lambda-mask 10.0 \
    || echo "plot failed"

echo
echo "=== Eval metrics ==="
python3 -m sim2real.scripts.eval_ckpt \
    --ckpt "$RUN_DIR/ckpts/$LAST_CKPT" \
    --sim many_cells --n-max 48 \
    --out "$RUN_DIR/eval_metrics.json" || echo "eval failed"

date
echo "DONE: $RUN_DIR"
