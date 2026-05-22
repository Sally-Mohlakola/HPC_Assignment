#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NP="${NP:-5}"
METRICS_ROOT="${RUN_OUTPUT_DIR:-${PROJECT_DIR}/metrics}"
COUNTER="${METRICS_ROOT}/.counter"

cd "$PROJECT_DIR"

make all

mkdir -p "$METRICS_ROOT"

CURRENT_COUNTER="$(cat "$COUNTER" 2>/dev/null || echo 0)"
if [[ ! "$CURRENT_COUNTER" =~ ^[0-9]+$ ]]; then
    echo "Invalid counter value in ${COUNTER}: ${CURRENT_COUNTER}" >&2
    exit 1
fi

RUN_COUNTER=$((CURRENT_COUNTER + 1))
echo "$RUN_COUNTER" > "$COUNTER"
SWEEP_ID="$RUN_COUNTER"

SINGLE_DIR="${METRICS_ROOT}/single_${RUN_COUNTER}"
LOG_DIR="${SINGLE_DIR}/logs"
CENTRALISED_OUTPUT_ROOT="${CENTRALISED_OUTPUT_ROOT:-${SINGLE_DIR}/centralised}"
FEDERATED_OUTPUT_ROOT="${FEDERATED_OUTPUT_ROOT:-${SINGLE_DIR}/federated}"
CENTRALISED_RUN_ID="${CENTRALISED_RUN_ID:-centralised}"
FEDERATED_RUN_ID="${FEDERATED_RUN_ID:-federated_np${NP}}"

mkdir -p "$CENTRALISED_OUTPUT_ROOT" "$FEDERATED_OUTPUT_ROOT" "$LOG_DIR"

echo "[run] project root: ${PROJECT_DIR}"
echo "[run] metrics dir:  ${SINGLE_DIR}"

echo
echo "Running centralised..."
echo "Output: ${CENTRALISED_OUTPUT_ROOT}"
(
    cd src
    export CENTRALISED_OUTPUT_ROOT
    export CENTRALISED_SWEEP_ID="${SWEEP_ID}"
    export CENTRALISED_RUN_ID
    ./centralised
) 2>&1 | tee "${LOG_DIR}/${CENTRALISED_RUN_ID}.log"

echo
echo "Running federated with ${NP} MPI processes..."
echo "Output: ${FEDERATED_OUTPUT_ROOT}"
(
    cd src
    export FEDERATED_OUTPUT_ROOT
    export FEDERATED_SWEEP_ID="${SWEEP_ID}"
    export FEDERATED_RUN_ID
    mpirun -np "${NP}" ./federated
) 2>&1 | tee "${LOG_DIR}/${FEDERATED_RUN_ID}.log"

echo
echo "Aggregating summaries..."
bash "${PROJECT_DIR}/bench/aggregate.sh" "$SINGLE_DIR"

echo
echo "Single run complete."
echo "Results: ${SINGLE_DIR}"
echo "Logs:    ${LOG_DIR}"
