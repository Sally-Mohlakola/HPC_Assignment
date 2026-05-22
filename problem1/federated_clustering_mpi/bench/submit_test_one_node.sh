#!/bin/bash
set -euo pipefail

# Submit the one-node sweep to Slurm.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_SCRIPT="${PROJECT_DIR}/bench/test_one_node.sh"
LOG_DIR="${PROJECT_DIR}/metrics/slurm"

if [[ ! -x "${BENCH_SCRIPT}" ]]; then
    echo "Benchmark script is not executable: ${BENCH_SCRIPT}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

export REPS="${REPS:-3}"
export NPS="${NPS:-3 5 7 9 13 16}"
export CURVE_NPS="${CURVE_NPS:-3 9 16}"

PARTITION="${SLURM_PARTITION:-stampede}"
TIME_LIMIT="${SLURM_TIME:-02:00:00}"
TASKS_PER_NODE="${SLURM_TASKS_PER_NODE:-}"
if [[ -z "${TASKS_PER_NODE}" ]]; then
    TASKS_PER_NODE=1
    for np in ${NPS}; do
        if (( np > TASKS_PER_NODE )); then
            TASKS_PER_NODE="${np}"
        fi
    done
fi

echo "[submit] project:        ${PROJECT_DIR}"
echo "[submit] script:         ${BENCH_SCRIPT}"
echo "[submit] logs:           ${LOG_DIR}/one_node_%x_%j.{out,err}"
echo "[submit] partition:      ${PARTITION}"
echo "[submit] time:           ${TIME_LIMIT}"
echo "[submit] tasks/node:     ${TASKS_PER_NODE}"
echo "[submit] reps:           ${REPS}"
echo "[submit] nps:            ${NPS}"
echo "[submit] curve nps:      ${CURVE_NPS}"

sbatch \
    --partition="${PARTITION}" \
    --nodes=1 \
    --ntasks-per-node="${TASKS_PER_NODE}" \
    --cpus-per-task=1 \
    --time="${TIME_LIMIT}" \
    --job-name=one_node_sweep \
    --chdir="${PROJECT_DIR}" \
    --output="${LOG_DIR}/one_node_%x_%j.out" \
    --error="${LOG_DIR}/one_node_%x_%j.err" \
    --export=ALL \
    --wrap="bash '${BENCH_SCRIPT}'"
