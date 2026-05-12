#!/bin/bash
#SBATCH --job-name=fed_mnist_10w
#SBATCH --partition=stampede
#SBATCH --nodes=2
#SBATCH --ntasks=11
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=fed_mnist_%j.out
#SBATCH --error=fed_mnist_%j.err

# Rank 0 is the federated server. Ranks 1-10 are workers.
# With block rank placement this gives:
#   node 1: rank 0 server + 5 workers
#   node 2: 5 workers

set -euo pipefail

WORKERS_PER_NODE=5
SERVER_RANKS=1
TOTAL_WORKERS=$((SLURM_JOB_NUM_NODES * WORKERS_PER_NODE))
TOTAL_TASKS=$((TOTAL_WORKERS + SERVER_RANKS))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p figures

echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "MPI ranks: ${TOTAL_TASKS} (${TOTAL_WORKERS} workers + ${SERVER_RANKS} server)"
echo "Workers per node: ${WORKERS_PER_NODE}"

if [[ "${SLURM_NTASKS}" -ne "${TOTAL_TASKS}" ]]; then
    echo "Expected ${TOTAL_TASKS} MPI ranks, but Slurm allocated ${SLURM_NTASKS}." >&2
    exit 1
fi

cd src

# If your Stampede environment requires explicit modules, load them before sbatch
# or uncomment and adjust these lines for the modules available on the cluster:
# module purge
# module load gcc openmpi

if ! command -v mpicxx >/dev/null 2>&1; then
    echo "mpicxx was not found. Load the MPI/compiler module for Stampede, then resubmit." >&2
    exit 1
fi

mpicxx -O3 -std=c++17 federated_main.cpp -o federated

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

srun --ntasks="${TOTAL_TASKS}" --distribution=block:block ./federated
