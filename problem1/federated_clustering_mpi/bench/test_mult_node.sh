#!/bin/bash
set -euo pipefail

# Multi-node sweep for Problem 1.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="${PROJECT_DIR}/bench"
cd "$PROJECT_DIR"

# Sweep settings can be overridden with env vars.
REPS="${REPS:-3}"

# Cluster cap is 16 ranks per node.
MAX_PPN=16

# Strategy A varies nodes at fixed PPN.
PPN_FIXED="${PPN_FIXED:-8}"
NODES_A=(${NODES_A:-2 3 4})

# Strategy B varies nodes at fixed total ranks.
TOTAL_B="${TOTAL_B:-48}"
NODES_B=(${NODES_B:-3 4 6 8})

# Compile-time data split.
FEDERATED_DEFINES="${FEDERATED_DEFINES:--DLABEL_SHARD_NONIID}"
BENCH_CXXFLAGS="${CXXFLAGS:--O3} -std=c++17"

# Toggle which strategies to run.
RUN_STRATEGY_A="${RUN_STRATEGY_A:-1}"
RUN_STRATEGY_B="${RUN_STRATEGY_B:-1}"

# Optional sbatch passthroughs.
SBATCH_PARTITION="${SBATCH_PARTITION:-stampede}"
SBATCH_TIME="${SBATCH_TIME:-02:00:00}"

# Allocate the next sweep id.
mkdir -p "$PROJECT_DIR/metrics"
COUNTER="$PROJECT_DIR/metrics/.counter"
CURRENT_COUNTER="$(cat "$COUNTER" 2>/dev/null || echo 0)"
if [[ ! "$CURRENT_COUNTER" =~ ^[0-9]+$ ]]; then
    echo "[config] invalid counter value in ${COUNTER}: ${CURRENT_COUNTER}" >&2
    exit 1
fi
SWEEP_ID=$((CURRENT_COUNTER + 1))
echo "$SWEEP_ID" > "$COUNTER"

SWEEP_DIR="${PROJECT_DIR}/metrics/sweep_${SWEEP_ID}"
mkdir -p "${SWEEP_DIR}/federated" "${SWEEP_DIR}/logs" "${SWEEP_DIR}/plots"

# Build once for all jobs.
echo "[build] federated  defines=${FEDERATED_DEFINES}  cxxflags=${BENCH_CXXFLAGS}"
make clean
make federated CXXFLAGS="${BENCH_CXXFLAGS}" FEDERATED_DEFINES="${FEDERATED_DEFINES}"

# Job ids for the aggregate dependency.
JOB_IDS=()

submit_job() {
    local strategy="$1"
    local nodes="$2"
    local ppn="$3"
    local total=$((nodes * ppn))
    local jobname="fed_${strategy}_n${nodes}_p${ppn}"

    if (( ppn > MAX_PPN )); then
        echo "[skip] strategy=${strategy} nodes=${nodes} ppn=${ppn} exceeds MAX_PPN=${MAX_PPN}"
        return 0
    fi

    echo
    echo "[sbatch] strategy=${strategy}  nodes=${nodes}  ntasks-per-node=${ppn}  total=${total}"

    local jid
    jid="$(sbatch --parsable \
        --partition="${SBATCH_PARTITION}" \
        --time="${SBATCH_TIME}" \
        --nodes="${nodes}" \
        --ntasks="${total}" \
        --ntasks-per-node="${ppn}" \
        --job-name="${jobname}" \
        --output="${SWEEP_DIR}/logs/${jobname}_%j.out" \
        --error="${SWEEP_DIR}/logs/${jobname}_%j.err" \
        --export=ALL,REPS="${REPS}",STRATEGY_NAME="${strategy}",SWEEP_ID="${SWEEP_ID}",SWEEP_DIR="${SWEEP_DIR}" \
        "${BENCH_DIR}/mult_node.slurm")"
    JOB_IDS+=("${jid}")
    echo "         submitted job ${jid}"
}

# Strategy A: fixed PPN.
if [[ "${RUN_STRATEGY_A}" == "1" ]]; then
    echo
    echo "============================================================"
    echo " Strategy A: fixed ppn=${PPN_FIXED}, nodes=${NODES_A[*]}"
    echo "============================================================"
    for n in "${NODES_A[@]}"; do
        submit_job "fixed_ppn" "${n}" "${PPN_FIXED}"
    done
fi

# Strategy B: fixed total ranks.
if [[ "${RUN_STRATEGY_B}" == "1" ]]; then
    echo
    echo "============================================================"
    echo " Strategy B: fixed total=${TOTAL_B}, nodes=${NODES_B[*]}"
    echo "============================================================"
    for n in "${NODES_B[@]}"; do
        if (( TOTAL_B % n != 0 )); then
            echo "[skip] nodes=${n} does not evenly divide total=${TOTAL_B}"
            continue
        fi
        ppn=$((TOTAL_B / n))
        submit_job "fixed_total" "${n}" "${ppn}"
    done
fi

if [[ ${#JOB_IDS[@]} -eq 0 ]]; then
    echo
    echo "[done] no jobs submitted (all strategies disabled or every config skipped)."
    exit 0
fi

# Aggregate after all jobs finish.
DEP="$(IFS=:; echo "${JOB_IDS[*]}")"
AGG_JID="$(sbatch --parsable \
    --partition="${SBATCH_PARTITION}" \
    --time="00:10:00" \
    --job-name="fed_aggregate" \
    --output="${SWEEP_DIR}/logs/fed_aggregate_%j.out" \
    --error="${SWEEP_DIR}/logs/fed_aggregate_%j.err" \
    --dependency="afterany:${DEP}" \
    --wrap "cd '${PROJECT_DIR}' && bench/aggregate.sh '${SWEEP_DIR}'")"

echo
echo "[done] submitted ${#JOB_IDS[@]} sweep job(s) + aggregate job ${AGG_JID}."
echo "       sweep id:   ${SWEEP_ID}"
echo "       sweep dir:  ${SWEEP_DIR}"
echo "       track with: squeue -u \$USER"
echo
echo "       The aggregate job runs automatically when the sweep finishes and"
echo "       builds ${SWEEP_DIR}/summary_multinode.csv. Then plot locally"
echo "       (plotting needs matplotlib, which the cluster nodes do not have):"
echo
echo "         python3 bench/plot_multi_node.py --project-dir ${PROJECT_DIR} \\"
echo "             --sweep-id ${SWEEP_ID} --reps ${REPS} \\"
echo "             --fixed-ppn ${PPN_FIXED} --fixed-total ${TOTAL_B}"
