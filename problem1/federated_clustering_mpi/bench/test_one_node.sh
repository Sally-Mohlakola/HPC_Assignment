#!/bin/bash
set -euo pipefail

# One-node sweep for Problem 1.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="${PROJECT_DIR}/bench"
cd "$PROJECT_DIR"

REPS="${REPS:-3}"
NPS=(${NPS:-3 5 7 9 13 16})
CURVE_NPS=(${CURVE_NPS:-3 9 16})
BENCH_CXXFLAGS="${CXXFLAGS:--O3} -std=c++17"

# Data splits and matching compile flags.
DIST_NAMES=(
    "round_robin_iid"
    "label_shard_noniid"
    "label_shard_noniid_rotate_feature_skew"
)
DIST_DEFINES=(
    "-DROUND_ROBIN_BASELINE"
    "-DLABEL_SHARD_NONIID"
    "-DLABEL_SHARD_NONIID -DROTATE_FEATURE_SKEW"
)

if [[ ${#DIST_NAMES[@]} -ne ${#DIST_DEFINES[@]} ]]; then
    echo "[config] DIST_NAMES and DIST_DEFINES must have the same length" >&2
    exit 1
fi

for curve_np in "${CURVE_NPS[@]}"; do
    found=0
    for np in "${NPS[@]}"; do
        if [[ "$curve_np" == "$np" ]]; then
            found=1
            break
        fi
    done
    if [[ "$found" == "0" ]]; then
        echo "[config] CURVE_NPS value ${curve_np} is not in NPS (${NPS[*]})" >&2
        exit 1
    fi
done

# Allocate the next sweep id.
mkdir -p metrics
COUNTER="metrics/.counter"
CURRENT_COUNTER="$(cat "$COUNTER" 2>/dev/null || echo 0)"
if [[ ! "$CURRENT_COUNTER" =~ ^[0-9]+$ ]]; then
    echo "[config] invalid counter value in ${COUNTER}: ${CURRENT_COUNTER}" >&2
    exit 1
fi
SWEEP_ID=$((CURRENT_COUNTER + 1))
echo "$SWEEP_ID" > "$COUNTER"

SWEEP_DIR="${PROJECT_DIR}/metrics/sweep_${SWEEP_ID}"
mkdir -p "${SWEEP_DIR}/centralised" "${SWEEP_DIR}/federated" \
         "${SWEEP_DIR}/logs" "${SWEEP_DIR}/plots"

echo "============================================================"
echo " Problem 1 one-node sweep"
echo " sweep id: ${SWEEP_ID}   folder: ${SWEEP_DIR}"
echo " reps: ${REPS}   nps: ${NPS[*]}"
echo " curve nps: ${CURVE_NPS[*]}"
echo " dists: ${DIST_NAMES[*]}"
echo " cxxflags: ${BENCH_CXXFLAGS}"
echo "============================================================"

# Phase 1: run the benchmark.
echo
echo "[phase 1/3] RUN"
echo "[build] centralised"
make centralised CXXFLAGS="${BENCH_CXXFLAGS}"

echo
echo "--- Centralised baseline: ${REPS} repetitions ---"
for rep in $(seq 1 "${REPS}"); do
    echo "[centralised] rep ${rep}/${REPS}"
    (
        cd src
        CENTRALISED_OUTPUT_ROOT="${SWEEP_DIR}/centralised" \
        CENTRALISED_SWEEP_ID="${SWEEP_ID}" \
        CENTRALISED_RUN_ID="central_rep${rep}" \
            ./centralised
    ) 2>&1 | tee "${SWEEP_DIR}/logs/central_rep${rep}.log"
done

for d_idx in "${!DIST_NAMES[@]}"; do
    dist_name="${DIST_NAMES[$d_idx]}"
    dist_define="${DIST_DEFINES[$d_idx]}"

    echo
    echo "--- Federated build: ${dist_name} (${dist_define}) ---"
    make clean
    make federated CXXFLAGS="${BENCH_CXXFLAGS}" FEDERATED_DEFINES="${dist_define}"

    for np in "${NPS[@]}"; do
        for rep in $(seq 1 "${REPS}"); do
            echo "[federated] dist=${dist_name} np=${np} rep=${rep}/${REPS}"
            (
                cd src
                FEDERATED_OUTPUT_ROOT="${SWEEP_DIR}/federated" \
                FEDERATED_SWEEP_ID="${SWEEP_ID}" \
                FEDERATED_RUN_ID="${dist_name}_np${np}_rep${rep}" \
                    mpirun -np "${np}" ./federated
            ) 2>&1 | tee "${SWEEP_DIR}/logs/${dist_name}_np${np}_rep${rep}.log"
        done
    done
done

# Phase 2: aggregate results.
echo
echo "[phase 2/3] AGGREGATE"
"${BENCH_DIR}/aggregate.sh" "${SWEEP_DIR}"

# Phase 3: plot results.
echo
echo "[phase 3/3] PLOT"
python3 "${BENCH_DIR}/plot_one_node.py" \
    --project-dir "${PROJECT_DIR}" \
    --sweep-id "${SWEEP_ID}" \
    --reps "${REPS}" \
    --nps "${NPS[@]}" \
    --curve-nps "${CURVE_NPS[@]}" \
    --dists "${DIST_NAMES[@]}"

echo
echo "[done] one-node sweep ${SWEEP_ID}"
echo "       results -> ${SWEEP_DIR}/"
echo "       plots   -> ${SWEEP_DIR}/plots/"
