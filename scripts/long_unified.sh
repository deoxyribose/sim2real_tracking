#!/usr/bin/env bash
# Long pretrain run with the full unified architecture:
#   - 5-dim affine z_where (sx, sy, theta, tx, ty)
#   - bottleneck bg (z_style=4, bg_base_res=4, bg_channels=(8,))
#   - GroupedDecoder with K=N_max enumerated groups
#   - supervised group identity (L_group) + temporal coherence (L_group_temp)
#   - glimpse_size=32 (handles elongated shapes)
#   - CoordConv, unified mask channel, stop_grad on recon path
#   - teacher z_where OFF (discovery test), teacher z_pres ON (gate help)
#   - frame curriculum T=3->12 over 25k, NaN guard, grad clip 1.0
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

SIM=${SIM:-flagella}
RUN_DIR=runs/unified_${SIM}
STEPS=${STEPS:-100000}
BATCH=${BATCH:-4}
LR=${LR:-2e-4}
N_MAX=${N_MAX:-8}

echo "=== unified pretrain ==="
echo "sim=$SIM  steps=$STEPS  batch=$BATCH  lr=$LR  n_max=$N_MAX  run_dir=$RUN_DIR"
date

python3 -m sim2real.scripts.pretrain \
    --sim "$SIM" \
    --steps "$STEPS" \
    --batch "$BATCH" \
    --lr "$LR" \
    --n-max "$N_MAX" \
    --teacher-force-zpres \
    --t-curriculum --t-start 3 --t-curriculum-steps 25000 \
    --n-groups "$N_MAX" \
    --lambda-group 1.0 \
    --lambda-group-temp 0.5 \
    --glimpse-size 32 \
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
    --sim "$SIM" --n-max "$N_MAX" \
    --out "$RUN_DIR/final_viz.png" || echo "viz failed"

echo "=== Eval metrics ==="
python3 -m sim2real.scripts.eval_ckpt \
    --ckpt "$RUN_DIR/ckpts/$LAST_CKPT" \
    --sim "$SIM" --n-max "$N_MAX" \
    --out "$RUN_DIR/eval_metrics.json" || echo "eval failed"

date
echo "DONE: $RUN_DIR"
