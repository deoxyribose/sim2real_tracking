#!/usr/bin/env bash
# Master pipeline: pretrain all 4 sims with the full unified architecture +
# glimpse-space mask supervision, then eval each. Runs sequentially — total
# wall time ~10h. Skips runs that already have a final checkpoint, so safe to
# restart.
#
# Triggered while the existing runs/unified_many_cells_v2 is still running.
# We wait for that to finish (presence of eval_metrics.json), then proceed
# through flagella → multiscale → worms.

set -uo pipefail   # don't -e: keep going even if one sim errors
cd "$(dirname "$0")/.."
export PYTHONPATH=.

LOG=runs/scheduled/master.log
mkdir -p runs/scheduled
echo "=== Master pipeline started ===" > "$LOG"
date >> "$LOG"

# Wait for the v2 cells run to finish.
echo "Waiting for runs/unified_many_cells_v2/eval_metrics.json..." | tee -a "$LOG"
until [ -f runs/unified_many_cells_v2/eval_metrics.json ]; do
    sleep 120
done
echo "many_cells_v2 done at $(date)" | tee -a "$LOG"

# Pretrain config per sim (n_max, n_groups, glimpse_size, n_transformer_layers).
# Grouped decoder only on flagella/worms where instance identity matters.
SIMS=(flagella multiscale worms)
declare -A NMAX=([flagella]=8 [multiscale]=24 [worms]=12)
declare -A NGROUPS=([flagella]=8 [multiscale]=1 [worms]=12)
declare -A GLIMPSE=([flagella]=32 [multiscale]=24 [worms]=32)

STEPS=100000
BATCH=4
LR=2e-4

for sim in "${SIMS[@]}"; do
    n_max=${NMAX[$sim]}
    n_groups=${NGROUPS[$sim]}
    glimpse=${GLIMPSE[$sim]}
    run_dir=runs/scheduled/${sim}_v2

    if [ -f "$run_dir/eval_metrics.json" ]; then
        echo "[SKIP] $sim already done" | tee -a "$LOG"
        continue
    fi
    mkdir -p "$run_dir"
    echo "=== $(date) === pretrain $sim (n_max=$n_max n_groups=$n_groups glimpse=$glimpse)" | tee -a "$LOG"

    GROUP_ARGS=""
    if [ "$n_groups" -gt 1 ]; then
        GROUP_ARGS="--n-groups $n_groups --lambda-group 1.0 --lambda-group-temp 0.5"
    fi

    python3 -m sim2real.scripts.pretrain \
        --sim "$sim" \
        --steps "$STEPS" \
        --batch "$BATCH" \
        --lr "$LR" \
        --n-max "$n_max" \
        --teacher-force-zpres \
        --t-curriculum --t-start 3 --t-curriculum-steps 20000 \
        --lambda-mask 0.0 \
        --lambda-mask-glimpse 10.0 \
        --lambda-recon 2.0 \
        --lambda-where 1.0 \
        --lambda-pres 1.0 \
        --lambda-aux 0.5 \
        --glimpse-size "$glimpse" \
        --n-transformer-layers 3 \
        $GROUP_ARGS \
        --log-every 250 \
        --ckpt-every 10000 \
        --run-dir "$run_dir" \
        > "$run_dir/run.log" 2>&1 \
        || echo "[FAIL] $sim pretrain exit code $?" | tee -a "$LOG"

    LAST_CKPT=$(ls -1 "$run_dir"/ckpts/ 2>/dev/null | sort -V | tail -n 1)
    if [ -z "$LAST_CKPT" ]; then
        echo "[FAIL] $sim has no checkpoints — skipping eval" | tee -a "$LOG"
        continue
    fi
    echo "$sim final ckpt: $LAST_CKPT" | tee -a "$LOG"

    python3 -m sim2real.scripts.viz_ckpt \
        --ckpt "$run_dir/ckpts/$LAST_CKPT" \
        --sim "$sim" --n-max "$n_max" \
        --n-groups "$n_groups" --glimpse-size "$glimpse" --n-transformer-layers 3 \
        --out "$run_dir/final_viz.png" 2>>"$run_dir/run.log" \
        || echo "[FAIL] $sim viz" | tee -a "$LOG"

    python3 -m sim2real.scripts.eval_ckpt \
        --ckpt "$run_dir/ckpts/$LAST_CKPT" \
        --sim "$sim" --n-max "$n_max" \
        --n-groups "$n_groups" --glimpse-size "$glimpse" --n-transformer-layers 3 \
        --out "$run_dir/eval_metrics.json" 2>>"$run_dir/run.log" \
        || echo "[FAIL] $sim eval" | tee -a "$LOG"

    echo "=== $(date) === done $sim" | tee -a "$LOG"
done

# Final summary across all sims (including the original v2 cells run).
echo "" | tee -a "$LOG"
echo "=== summary ===" | tee -a "$LOG"
python3 -c "
import json, os, glob
runs = []
runs += sorted(glob.glob('runs/unified_many_cells_v2/eval_metrics.json'))
runs += sorted(glob.glob('runs/scheduled/*_v2/eval_metrics.json'))
print('%-14s %8s %8s %8s %8s %8s' % ('sim', 'PSNR', 'SSIM', 'IoU', 'sil', 'id_sw'))
for r in runs:
    try:
        with open(r) as f: m = json.load(f)
    except Exception:
        continue
    parts = r.split('/')
    label = parts[-2].replace('_v2', '').replace('unified_many_cells', 'many_cells')
    print('%-14s %8.2f %8.3f %8.3f %8.3f %8d' % (
        label, m.get('psnr', float('nan')), m.get('ssim', float('nan')),
        m.get('seg_iou', float('nan')), m.get('silhouette_zwhat', float('nan')),
        m.get('id_switches_total', -1),
    ))
" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "ALL DONE at $(date)" | tee -a "$LOG"
