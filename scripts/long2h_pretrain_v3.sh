#!/usr/bin/env bash
# ~2h pretrain run on many_cells with everything the literature/cellulose suggested:
#   - teacher forcing on z_where and z_pres (model uses GT[t-1] as residual anchor and
#     gates propagate/discover with GT presence; predicted z_where/z_pres still trained)
#   - SQAIR-style frame curriculum (T = 3 → 12 over the first 25k steps)
#   - stop_gradient on z_where + z_pres before stn_write (cellulose-style: recon trains
#     decoder + z_what only; pose head trained exclusively by L_where)
#   - NaN-guard: gradient finite-check inside the jitted step
#   - grad clip 1.0, AdamW with cosine LR
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

RUN_DIR=runs/pretrain_many_cells_2h_v3
STEPS=${STEPS:-100000}
BATCH=${BATCH:-4}
LR=${LR:-2e-4}

echo "=== Starting 2h pretrain ==="
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
    --lambda-mask 3.0 \
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
    --lambda-recon 2.0 --lambda-where 1.0 --lambda-pres 1.0 --lambda-mask 3.0 \
    || echo "plot failed"

echo
echo "=== Eval metrics ==="
python3 -m sim2real.scripts.eval_ckpt \
    --ckpt "$RUN_DIR/ckpts/$LAST_CKPT" \
    --sim many_cells --n-max 48 \
    --out "$RUN_DIR/eval_metrics.json" || echo "eval failed"

date
echo "DONE: $RUN_DIR"
