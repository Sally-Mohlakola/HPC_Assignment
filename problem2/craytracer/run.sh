#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_ROOT="${SCRIPT_DIR}/output"
METRICS_ROOT="${SCRIPT_DIR}/metrics"
COUNTER="${METRICS_ROOT}/.counter"

mkdir -p "$OUTPUT_ROOT" "$METRICS_ROOT" renders
make

CURRENT_COUNTER="$(cat "$COUNTER" 2>/dev/null || echo 0)"
if [[ ! "$CURRENT_COUNTER" =~ ^[0-9]+$ ]]; then
    echo "Invalid counter value in ${COUNTER}: ${CURRENT_COUNTER}" >&2
    exit 1
fi

RUN_COUNTER=$((CURRENT_COUNTER + 1))
echo "$RUN_COUNTER" > "$COUNTER"
SWEEP_ID="$RUN_COUNTER"

SINGLE_DIR="${METRICS_ROOT}/single_${RUN_COUNTER}"
mkdir -p "$SINGLE_DIR"

export CRAYTRACER_METRICS_DIR="$SINGLE_DIR"
export CRAYTRACER_SWEEP_ID="$SWEEP_ID"

echo "[run] project root: ${SCRIPT_DIR}"
echo "[run] output dir:   ${OUTPUT_ROOT}"
echo "[run] metrics dir:  ${SINGLE_DIR}"


./renders/raytracer "$@" 2>&1 | tee "${SINGLE_DIR}/run.log"

echo
echo "Single run complete."
echo "Images:  ${OUTPUT_ROOT}"
echo "Metrics: ${SINGLE_DIR}/summary.csv"
echo "Log:     ${SINGLE_DIR}/run.log"
