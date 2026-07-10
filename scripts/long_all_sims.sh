#!/usr/bin/env bash
# 200k-step pretrain on each of 4 sims with the Exp 13 recipe. Six runs total:
#   - 4 "matched-old" runs: many_cells (grp=1), flagella (grp=8), multiscale (grp=1), worms (grp=12)
#   - 2 "no-groups ablation" runs: flagella (grp=1), worms (grp=1)
#
# Ordered shortest-first so we see results incrementally. Idempotent: skips any
# sim whose eval_metrics.json already exists.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=/home/frans/jaxup_venv/bin/python

BASE_DIR=runs/long_all_sims
LOG="$BASE_DIR/master.log"
mkdir -p "$BASE_DIR"
echo "=== long pipeline started $(date) ===" > "$LOG"

STEPS=200000
BATCH=4
LR=2e-4

# Per-run definitions: name  sim  n_max  glimpse  n_groups
RUNS=(
    "flagella_grp1     flagella   8   32  1"
    "flagella_grp8     flagella   8   32  8"
    "worms_grp1        worms     12   32  1"
    "worms_grp12       worms     12   32 12"
    "multiscale        multiscale 24  24  1"
    "many_cells        many_cells 48  16  1"
)

for spec in "${RUNS[@]}"; do
    read -r name sim n_max glimpse n_groups <<<"$spec"
    run_dir="$BASE_DIR/$name"

    if [ -f "$run_dir/eval_metrics.json" ]; then
        echo "[SKIP] $name already done" | tee -a "$LOG"
        continue
    fi
    mkdir -p "$run_dir"
    echo "" | tee -a "$LOG"
    echo "=== $(date) === pretrain $name (sim=$sim n_max=$n_max glimpse=$glimpse n_groups=$n_groups)" | tee -a "$LOG"

    GROUP_ARGS=""
    if [ "$n_groups" -gt 1 ]; then
        GROUP_ARGS="--n-groups $n_groups --lambda-group 1.0 --lambda-group-temp 0.5"
    fi

    $PY -m sim2real.scripts.pretrain \
        --sim "$sim" --steps "$STEPS" --batch "$BATCH" --lr "$LR" \
        --n-max "$n_max" \
        --t-curriculum --t-start 3 --t-curriculum-steps 20000 \
        --glimpse-size "$glimpse" \
        --n-transformer-layers 3 \
        --stem-strides 2 2 1 \
        --lambda-recon 2.0 \
        --lambda-where 5.0 \
        --lambda-pres 1.0 \
        --lambda-mask 0.0 \
        --lambda-mask-glimpse 10.0 \
        --lambda-appear-glimpse 10.0 \
        --lambda-slot-contrast 1.0 \
        --slot-contrast-tau 0.1 \
        --lambda-aux 0.5 \
        --matching-mode per_frame \
        --match-once-after 500 \
        --log-every 500 \
        --ckpt-every 5000 \
        --eval-every 1000 \
        --eval-batch 4 \
        $GROUP_ARGS \
        --run-dir "$run_dir" \
        > "$run_dir/run.log" 2>&1 \
        || echo "[FAIL] $name pretrain exit $?" | tee -a "$LOG"

    LAST_CKPT=$(ls -1 "$run_dir/ckpts/" 2>/dev/null | sort -V | tail -n 1)
    if [ -z "$LAST_CKPT" ]; then
        echo "[FAIL] $name no ckpts" | tee -a "$LOG"
        continue
    fi
    echo "$name last ckpt: $LAST_CKPT" | tee -a "$LOG"

    $PY -m sim2real.scripts.viz_ckpt \
        --ckpt "$run_dir/ckpts/$LAST_CKPT" \
        --sim "$sim" --n-max "$n_max" \
        --glimpse-size "$glimpse" --n-transformer-layers 3 --stem-strides 2 2 1 \
        --n-groups "$n_groups" \
        --out "$run_dir/final_viz.png" 2>>"$run_dir/run.log" \
        || echo "[FAIL] $name viz" | tee -a "$LOG"

    $PY -m sim2real.scripts.eval_ckpt \
        --ckpt "$run_dir/ckpts/$LAST_CKPT" \
        --sim "$sim" --n-max "$n_max" \
        --glimpse-size "$glimpse" --n-transformer-layers 3 --stem-strides 2 2 1 \
        --n-groups "$n_groups" \
        --out "$run_dir/eval_metrics.json" 2>>"$run_dir/run.log" \
        || echo "[FAIL] $name eval" | tee -a "$LOG"

    echo "=== $(date) === done $name" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== summary ===" | tee -a "$LOG"
$PY -c "
import json, glob
runs = sorted(glob.glob('$BASE_DIR/*/eval_metrics.json'))
print('%-20s %8s %8s %8s %8s %8s' % ('run', 'PSNR', 'SSIM', 'IoU', 'sil', 'id_sw'))
for r in runs:
    try:
        with open(r) as f: m = json.load(f)
    except Exception:
        continue
    label = r.split('/')[-2]
    print('%-20s %8.2f %8.3f %8.3f %8.3f %8d' % (
        label, m.get('psnr', float('nan')), m.get('ssim', float('nan')),
        m.get('seg_iou', float('nan')), m.get('silhouette_zwhat', float('nan')),
        m.get('id_switches_total', -1),
    ))
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "ALL DONE at $(date)" | tee -a "$LOG"
