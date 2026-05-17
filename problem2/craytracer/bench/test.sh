#!/bin/bash
#SBATCH --partition=stampede
#SBATCH --cpus-per-task=4
#SBATCH --time=09:00:00
#SBATCH --job-name=craytracer_sweep
#SBATCH --output=metrics/slurm_%x_%j.out
#SBATCH --error=metrics/slurm_%x_%j.err
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

mkdir -p output metrics
make

BIN="./renders/raytracer"

BASE_BLOCK=256
BASE_DEPTH=50
BASE_SPHERES=200

#BLOCK_SIZES=(32 64 128 256 512 1024)
#SPHERE_COUNTS=(50 100 200 400 800)
#RAY_DEPTHS=(5 10 25 50 75 100)
RAY_DEPTHS=(75 100)

RUNS_PER_TEST=3

# ----- Allocate the next sweep id -----------------------------------------
# All generated files for this run live under metrics/sweep_<N>/.
COUNTER="metrics/.sweep_counter"
SWEEP_ID=$(( $(cat "$COUNTER" 2>/dev/null || echo 0) + 1 ))
echo "$SWEEP_ID" > "$COUNTER"

SWEEP_DIR="${PROJECT_ROOT}/metrics/sweep_${SWEEP_ID}"
LOG_DIR="${SWEEP_DIR}/logs"
PLOT_DIR="${SWEEP_DIR}/plots"
mkdir -p "$SWEEP_DIR" "$LOG_DIR" "$PLOT_DIR"

export CRAYTRACER_METRICS_DIR="$SWEEP_DIR"
export CRAYTRACER_SWEEP_ID="$SWEEP_ID"

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

#echo "###################################################################"
#echo "# Sweep 1: block size  (depth=$BASE_DEPTH, spheres=$BASE_SPHERES)"
#echo "# Each config run ${RUNS_PER_TEST}x"
#echo "###################################################################"
#for bs in "${BLOCK_SIZES[@]}"; do
#    run_repeated "block_${bs}" \
#        --block-size "$bs" \
#        --max-depth "$BASE_DEPTH" \
#        --num-spheres "$BASE_SPHERES"
#done

#echo "###################################################################"
#echo "# Sweep 2: object count  (block=$BASE_BLOCK, depth=$BASE_DEPTH)"
#echo "# Each config run ${RUNS_PER_TEST}x"
#echo "###################################################################"
#for n in "${SPHERE_COUNTS[@]}"; do
#    run_repeated "spheres_${n}" \
#        --block-size "$BASE_BLOCK" \
#        --max-depth "$BASE_DEPTH" \
#        --num-spheres "$n"
#done

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
echo "Aggregated CSV:  ${SWEEP_DIR}/summary.csv"

echo
echo "###################################################################"
echo "# Generating averaged plots for sweep ${SWEEP_ID}"
echo "###################################################################"
python3 "$PROJECT_ROOT/bench/plot_results.py" \
    --project-dir "$PROJECT_ROOT" \
    --sweep-id "$SWEEP_ID"
