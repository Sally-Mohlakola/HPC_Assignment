#!/bin/bash
set -euo pipefail

# Multi-node experiment sweep for Problem 1.
#
# Submits one sbatch per (nodes, ntasks-per-node) tuple, covering two
# strategies:
#
#   A) fixed PPN, varying nodes  — process count per node is held constant
#      while node count grows. Total ranks scale linearly with N.
#        e.g.  PPN_FIXED=10, NODES_A=(2 3 4)
#              -> --nodes=2 --ntasks-per-node=10 --ntasks=20
#                 --nodes=3 --ntasks-per-node=10 --ntasks=30
#                 --nodes=4 --ntasks-per-node=10 --ntasks=40
#
#   B) fixed total ranks, varying nodes — TOTAL_B is held constant and
#      split evenly across nodes. Only node counts that divide TOTAL_B
#      cleanly are kept (uneven splits produce non-uniform per-node loads
#      that complicate analysis).
#        e.g.  TOTAL_B=24, NODES_B=(2 3 4 6 8)
#              -> --nodes=2 --ntasks-per-node=12 --ntasks=24
#                 --nodes=3 --ntasks-per-node=8  --ntasks=24
#                 --nodes=4 --ntasks-per-node=6  --ntasks=24
#                 --nodes=6 --ntasks-per-node=4  --ntasks=24
#                 --nodes=8 --ntasks-per-node=3  --ntasks=24
#
# Rank 0 is the federated server (lives on node 0). All other ranks are
# workers. Output writes to the shared federated_metrics/ tree (same as the
# one-node sweep); every summary row is tagged with num_nodes so multi-node
# runs can be filtered apart from single-node ones at plot time.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Sweep configuration — override via env vars when invoking the script.
# ---------------------------------------------------------------------------
REPS="${REPS:-3}"

PPN_FIXED="${PPN_FIXED:-10}"
NODES_A=(${NODES_A:-2 3 4})

TOTAL_B="${TOTAL_B:-24}"
NODES_B=(${NODES_B:-2 3 4 6 8})

# Distribution baked into the binary at compile time (same default as
# test_one_node.sh so the two sweeps are comparable).
FEDERATED_DEFINES="${FEDERATED_DEFINES:--DLABEL_SHARD_NONIID}"

# Toggle which strategies to run.
RUN_STRATEGY_A="${RUN_STRATEGY_A:-1}"
RUN_STRATEGY_B="${RUN_STRATEGY_B:-1}"

# Optional sbatch passthroughs.
SBATCH_PARTITION="${SBATCH_PARTITION:-stampede}"
SBATCH_TIME="${SBATCH_TIME:-02:00:00}"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/federated_metrics"

# Stamped before any sbatch so plot_multi_node.py can filter to this sweep.
SWEEP_START="$(date +%Y-%m-%d_%H-%M-%S)"

# ---------------------------------------------------------------------------
# Build once. All sbatched jobs share src/federated.
# ---------------------------------------------------------------------------
echo "[build] federated  defines=${FEDERATED_DEFINES}"
make clean
make federated FEDERATED_DEFINES="${FEDERATED_DEFINES}"

submit_job() {
    local strategy="$1"
    local nodes="$2"
    local ppn="$3"
    local total=$((nodes * ppn))
    local jobname="fed_${strategy}_n${nodes}_p${ppn}"

    echo
    echo "[sbatch] strategy=${strategy}  nodes=${nodes}  ntasks-per-node=${ppn}  total=${total}"

    sbatch \
        --partition="${SBATCH_PARTITION}" \
        --time="${SBATCH_TIME}" \
        --nodes="${nodes}" \
        --ntasks="${total}" \
        --ntasks-per-node="${ppn}" \
        --job-name="${jobname}" \
        --export=ALL,REPS="${REPS}",STRATEGY_NAME="${strategy}" \
        "${PROJECT_DIR}/mult_node.slurm"
}

# ---------------------------------------------------------------------------
# Strategy A — fixed PPN, varying nodes.
# ---------------------------------------------------------------------------
if [[ "${RUN_STRATEGY_A}" == "1" ]]; then
    echo
    echo "============================================================"
    echo " Strategy A: fixed ppn=${PPN_FIXED}, nodes=${NODES_A[*]}"
    echo "============================================================"
    for n in "${NODES_A[@]}"; do
        submit_job "fixed_ppn" "${n}" "${PPN_FIXED}"
    done
fi

# ---------------------------------------------------------------------------
# Strategy B — fixed total, varying nodes (only divisors of TOTAL_B).
# ---------------------------------------------------------------------------
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

echo
echo "[done] all jobs submitted. Track with: squeue -u \$USER"
echo "       output -> federated_metrics/  (rows tagged with num_nodes; filter by it at plot time)"
echo
echo "       Once every job has finished, plot with:"
echo "         python3 plot_multi_node.py --project-dir ${PROJECT_DIR} \\"
echo "             --start ${SWEEP_START} --reps ${REPS} \\"
echo "             --fixed-ppn ${PPN_FIXED} --fixed-total ${TOTAL_B}"
