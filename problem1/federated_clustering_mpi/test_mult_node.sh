#!/bin/bash
set -euo pipefail

# Multi-node experiment sweep for Problem 1.
#
# Submits one sbatch per (nodes, ntasks-per-node) tuple, covering two
# strategies:
#
#   A) fixed PPN, varying nodes  — process count per node is held constant
#      while node count grows. Total ranks scale linearly with N.
#        e.g.  PPN_FIXED=8, NODES_A=(2 3 4)
#              -> --nodes=2 --ntasks-per-node=8 --ntasks=16
#                 --nodes=3 --ntasks-per-node=8 --ntasks=24
#                 --nodes=4 --ntasks-per-node=8 --ntasks=32
#
#   B) fixed total ranks, varying nodes — TOTAL_B is held constant and
#      split evenly across nodes. Only node counts that divide TOTAL_B
#      cleanly are kept (uneven splits produce non-uniform per-node loads
#      that complicate analysis).
#        e.g.  TOTAL_B=48, NODES_B=(3 4 6 8)
#              -> --nodes=3 --ntasks-per-node=16 --ntasks=48
#                 --nodes=4 --ntasks-per-node=12 --ntasks=48
#                 --nodes=6 --ntasks-per-node=8  --ntasks=48
#                 --nodes=8 --ntasks-per-node=6  --ntasks=48
#
#      The cluster caps tasks-per-node at 16 (one rank per physical core),
#      so configurations with PPN > 16 are skipped automatically.
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

# Cluster caps per-node task count at 16 (one rank per physical core on the
# Xeon E5-2680 nodes). Any (TOTAL_B, nodes) pair where TOTAL_B/nodes > 16
# will be rejected by sbatch ("CPU count per node can not be satisfied").
MAX_PPN=16

# Strategy A (weak scaling): PPN=8 pins one rank per physical core within a
# single NUMA socket (8 cores/socket, 2 sockets/node). This keeps each
# worker's L1/L2 private and gives a clean share of one L3 domain, so
# per-rank performance is constant as nodes are added.
PPN_FIXED="${PPN_FIXED:-8}"
NODES_A=(${NODES_A:-2 3 4})

# Strategy B (strong scaling): total=48 factors over every node count in
# NODES_B, giving high-utilisation per-node PPN values (16, 12, 8, 6) that
# all respect the 16-rank-per-node cluster cap. nodes=2 is intentionally
# excluded because TOTAL_B/2 = 24 > 16 and would be rejected by sbatch.
# 48 keeps per-rank work meaningful even at 8 nodes (6 ranks/node), whereas
# the earlier total=24 fell to 3 ranks/node — most cores idle and per-rank
# work too small to dominate communication cost.
TOTAL_B="${TOTAL_B:-48}"
NODES_B=(${NODES_B:-3 4 6 8})

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

    if (( ppn > MAX_PPN )); then
        echo "[skip] strategy=${strategy} nodes=${nodes} ppn=${ppn} exceeds MAX_PPN=${MAX_PPN}"
        return
    fi

    echo
    echo "[sbatch] strategy=${strategy}  nodes=${nodes}  ntasks-per-node=${ppn}  total=${total}"

    sbatch \
        --partition="${SBATCH_PARTITION}" \
        --time="${SBATCH_TIME}" \
        --nodes="${nodes}" \
        --ntasks="${total}" \
        --ntasks-per-node="${ppn}" \
        --job-name="${jobname}" \
        --output="${PROJECT_DIR}/logs/${jobname}_%j.out" \
        --error="${PROJECT_DIR}/logs/${jobname}_%j.err" \
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
