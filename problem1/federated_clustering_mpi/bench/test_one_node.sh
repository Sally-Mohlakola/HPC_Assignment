#!/bin/bash
set -euo pipefail

# One-node experiment sweep for Problem 1, run as three phases:
#
#   Phase 1  RUN        centralised baseline + federated runs across MPI
#                       process counts. Each run writes only its own private
#                       <kind>/<run_id>/ directory inside this sweep's folder
#                       -- no shared file is touched, so nothing can race.
#   Phase 2  AGGREGATE  aggregate.sh gathers each run_summary.csv into the
#                       sweep's summary sheets (summary_centralised.csv and
#                       summary_onenode.csv).
#   Phase 3  PLOT       plot_one_node.py renders charts into the sweep's plots/.
#
# Everything for the sweep lives under metrics/sweep_<N>/, where N is an
# incrementing counter kept in metrics/.sweep_counter.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="${PROJECT_DIR}/bench"
cd "$PROJECT_DIR"

REPS="${REPS:-3}"
NPS=(${NPS:-3 5 7 9 13 16})

# Single distribution requested for the sweep. Each entry is a name + the
# corresponding -D flags passed to mpicxx via FEDERATED_DEFINES.
DIST_NAMES=("label_shard_noniid")
DIST_DEFINES=("-DLABEL_SHARD_NONIID")

# ----- Allocate the next sweep id -----------------------------------------
# An incrementing counter; each sweep's entire contents live in its own folder.
mkdir -p metrics
COUNTER="metrics/.sweep_counter"
SWEEP_ID=$(( $(cat "$COUNTER" 2>/dev/null || echo 0) + 1 ))
echo "$SWEEP_ID" > "$COUNTER"

SWEEP_DIR="${PROJECT_DIR}/metrics/sweep_${SWEEP_ID}"
mkdir -p "${SWEEP_DIR}/centralised" "${SWEEP_DIR}/federated" \
         "${SWEEP_DIR}/logs" "${SWEEP_DIR}/plots"

echo "============================================================"
echo " Problem 1 one-node sweep"
echo " sweep id: ${SWEEP_ID}   folder: ${SWEEP_DIR}"
echo " reps: ${REPS}   nps: ${NPS[*]}"
echo "============================================================"

# ===========================================================================
# PHASE 1 -- RUN
#
# Build centralised once; federated is rebuilt per distribution because the
# flags are baked in at compile time. Each run is handed its sweep id and a
# run id unique within the sweep; the binary uses the run id as its output
# directory name, so runs never collide.
# ===========================================================================
echo
echo "[phase 1/3] RUN"
echo "[build] centralised"
make centralised

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
    make federated FEDERATED_DEFINES="${dist_define}"

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

# ===========================================================================
# PHASE 2 -- AGGREGATE
# ===========================================================================
echo
echo "[phase 2/3] AGGREGATE"
"${BENCH_DIR}/aggregate.sh" "${SWEEP_DIR}"

# ===========================================================================
# PHASE 3 -- PLOT
# ===========================================================================
echo
echo "[phase 3/3] PLOT"
python3 "${BENCH_DIR}/plot_one_node.py" \
    --project-dir "${PROJECT_DIR}" \
    --sweep-id "${SWEEP_ID}" \
    --reps "${REPS}" \
    --nps "${NPS[@]}" \
    --dists "${DIST_NAMES[@]}"

echo
echo "[done] one-node sweep ${SWEEP_ID}"
echo "       results -> ${SWEEP_DIR}/"
echo "       plots   -> ${SWEEP_DIR}/plots/"
