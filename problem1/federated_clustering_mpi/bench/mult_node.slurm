#!/bin/bash
#SBATCH --partition=stampede
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
# test_mult_node.sh usually sets log paths.
#SBATCH --output=metrics/mult_node_%x_%j.out
#SBATCH --error=metrics/mult_node_%x_%j.err

# Multi-node federated run payload.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

REPS="${REPS:-3}"
STRATEGY_NAME="${STRATEGY_NAME:-unspecified}"
# Fallback keeps direct sbatch runs under sweep_0.
SWEEP_ID="${SWEEP_ID:-0}"
SWEEP_DIR="${SWEEP_DIR:-${PROJECT_DIR}/metrics/sweep_${SWEEP_ID}}"

mkdir -p "${SWEEP_DIR}/federated"

echo "Job ID:        ${SLURM_JOB_ID}"
echo "Job name:      ${SLURM_JOB_NAME}"
echo "Strategy:      ${STRATEGY_NAME}"
echo "Nodes:         ${SLURM_JOB_NUM_NODES} (${SLURM_JOB_NODELIST})"
echo "Tasks total:   ${SLURM_NTASKS}"
echo "Tasks/node:    ${SLURM_NTASKS_PER_NODE:-?}"
echo "Sweep id:      ${SWEEP_ID}"
echo "Sweep dir:     ${SWEEP_DIR}"

if ! command -v mpicxx >/dev/null 2>&1; then
    echo "mpicxx not found. Load the MPI/compiler module before submitting." >&2
    exit 1
fi

# Binary is built by test_mult_node.sh.
if [[ ! -x "${PROJECT_DIR}/src/federated" ]]; then
    echo "src/federated not built. Run test_mult_node.sh (which builds once) before sbatching directly." >&2
    exit 1
fi

cd src

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export FEDERATED_SWEEP_ID="${SWEEP_ID}"
export FEDERATED_OUTPUT_ROOT="${SWEEP_DIR}/federated"
PPN="${SLURM_NTASKS_PER_NODE:-na}"

for rep in $(seq 1 "${REPS}"); do
    # Slurm job id keeps run folders unique.
    export FEDERATED_RUN_ID="${STRATEGY_NAME}_n${SLURM_JOB_NUM_NODES}_p${PPN}_rep${rep}_j${SLURM_JOB_ID}"
    echo
    echo "[mult_node] run_id=${FEDERATED_RUN_ID} rep=${rep}/${REPS}"
    srun --mpi=pmix -n "${SLURM_NTASKS}" ./federated || \
        mpirun -np "${SLURM_NTASKS}" --map-by ppr:"${SLURM_NTASKS_PER_NODE}":node ./federated
done
