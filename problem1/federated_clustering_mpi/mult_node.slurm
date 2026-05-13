#!/bin/bash
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/mult_node_%x_%j.out
#SBATCH --error=logs/mult_node_%x_%j.err

# Parameterised multi-node federated run. test_mult_node.sh submits this once
# per (nodes, ppn) tuple, overriding --nodes / --ntasks / --ntasks-per-node /
# --job-name on the sbatch command line and exporting REPS + STRATEGY_NAME.
#
# Output goes to the same ../federated_metrics/ tree as the one-node sweep;
# every summary.csv row carries num_nodes so multi-node runs are distinguishable
# from single-node runs after the fact.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_DIR"

REPS="${REPS:-3}"
STRATEGY_NAME="${STRATEGY_NAME:-unspecified}"

mkdir -p "${PROJECT_DIR}/federated_metrics"

echo "Job ID:        ${SLURM_JOB_ID}"
echo "Job name:      ${SLURM_JOB_NAME}"
echo "Strategy:      ${STRATEGY_NAME}"
echo "Nodes:         ${SLURM_JOB_NUM_NODES} (${SLURM_JOB_NODELIST})"
echo "Tasks total:   ${SLURM_NTASKS}"
echo "Tasks/node:    ${SLURM_NTASKS_PER_NODE:-?}"

if ! command -v mpicxx >/dev/null 2>&1; then
    echo "mpicxx not found. Load the MPI/compiler module before submitting." >&2
    exit 1
fi

# Binary is pre-built by test_mult_node.sh so every sbatch hits the same one.
if [[ ! -x "${PROJECT_DIR}/src/federated" ]]; then
    echo "src/federated not built. Run test_mult_node.sh (which builds once) before sbatching directly." >&2
    exit 1
fi

cd src

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

for rep in $(seq 1 "${REPS}"); do
    echo
    echo "[mult_node] strategy=${STRATEGY_NAME} nodes=${SLURM_JOB_NUM_NODES} ntasks=${SLURM_NTASKS} rep=${rep}/${REPS}"
    srun --mpi=pmix -n "${SLURM_NTASKS}" ./federated || \
        mpirun -np "${SLURM_NTASKS}" --map-by ppr:"${SLURM_NTASKS_PER_NODE}":node ./federated
    sleep 1
done
