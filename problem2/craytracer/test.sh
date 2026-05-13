#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p output metrics
make

BIN="./renders/raytracer"

BASE_BLOCK=256
BASE_DEPTH=50
BASE_SPHERES=200

BLOCK_SIZES=(32 64 128 256 512 1024)
SPHERE_COUNTS=(50 100 200 400 800)
RAY_DEPTHS=(5 10 25 50 75 100)

RUNS_PER_TEST=3

LOG_DIR="metrics/logs"
PLOT_DIR="metrics/plots"
mkdir -p "$LOG_DIR" "$PLOT_DIR"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"

run_one() {
    local label="$1"; shift
    local logfile="$LOG_DIR/${STAMP}_${label}.log"
    echo
    echo "==================================================================="
    echo "[$label] $BIN $*"
    echo "==================================================================="
    "$BIN" "$@" 2>&1 | tee "$logfile"
}

run_repeated() {
    local label="$1"; shift
    for r in $(seq 1 "$RUNS_PER_TEST"); do
        run_one "${label}_r${r}" "$@"
    done
}

echo "###################################################################"
echo "# Sweep 1: block size  (depth=$BASE_DEPTH, spheres=$BASE_SPHERES)"
echo "# Each config run ${RUNS_PER_TEST}x"
echo "###################################################################"
for bs in "${BLOCK_SIZES[@]}"; do
    run_repeated "block_${bs}" \
        --block-size "$bs" \
        --max-depth "$BASE_DEPTH" \
        --num-spheres "$BASE_SPHERES"
done

echo "###################################################################"
echo "# Sweep 2: object count  (block=$BASE_BLOCK, depth=$BASE_DEPTH)"
echo "# Each config run ${RUNS_PER_TEST}x"
echo "###################################################################"
for n in "${SPHERE_COUNTS[@]}"; do
    run_repeated "spheres_${n}" \
        --block-size "$BASE_BLOCK" \
        --max-depth "$BASE_DEPTH" \
        --num-spheres "$n"
done

echo "###################################################################"
echo "# Sweep 3: ray depth  (block=$BASE_BLOCK, spheres=$BASE_SPHERES)"
echo "# Each config run ${RUNS_PER_TEST}x"
echo "###################################################################"
for d in "${RAY_DEPTHS[@]}"; do
    run_repeated "depth_${d}" \
        --block-size "$BASE_BLOCK" \
        --max-depth "$d" \
        --num-spheres "$BASE_SPHERES"
done

echo
echo "All sweeps complete."
echo "Per-run logs:    $LOG_DIR/${STAMP}_*.log"
echo "Aggregated CSV:  metrics/summary.csv"

echo
echo "###################################################################"
echo "# Generating averaged plots from this session (since $STAMP)"
echo "###################################################################"
python3 "$SCRIPT_DIR/plot_results.py" \
    --csv "$SCRIPT_DIR/metrics/summary.csv" \
    --since "$STAMP" \
    --output-dir "$PLOT_DIR" \
    --stamp "$STAMP" \
    --base-block "$BASE_BLOCK" \
    --base-depth "$BASE_DEPTH" \
    --base-spheres "$BASE_SPHERES"
