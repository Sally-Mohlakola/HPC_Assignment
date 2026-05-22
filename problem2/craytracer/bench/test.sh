#!/usr/bin/env bash
set -euo pipefail

# Submit the craytracer sweep from the repo.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/craytracer_sweep.slurm"

case "$SCRIPT_DIR" in
    /var/lib/slurm/*|/var/spool/slurm/*|/run/slurm/*)
        echo "bench/test.sh was submitted to Slurm as the batch script." >&2
        echo "Run it directly instead: bench/test.sh" >&2
        exit 1
        ;;
esac

if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "Could not find Slurm payload script: ${SLURM_SCRIPT}" >&2
    echo "Run this wrapper from the checked-out repository, not from a copied Slurm script." >&2
    exit 1
fi

OUTPUT_DIR="${PROJECT_ROOT}/output"
METRICS_ROOT="${PROJECT_ROOT}/metrics"

if [[ ! -w "$PROJECT_ROOT" ]]; then
    echo "Project root is not writable: ${PROJECT_ROOT}" >&2
    exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch was not found in PATH. Run this wrapper on the Slurm login node." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$METRICS_ROOT"

if [[ ! -w "$OUTPUT_DIR" || ! -w "$METRICS_ROOT" ]]; then
    echo "Project directories are not writable:" >&2
    echo "  project: ${PROJECT_ROOT}" >&2
    echo "  output:  ${OUTPUT_DIR}" >&2
    echo "  metrics: ${METRICS_ROOT}" >&2
    exit 1
fi

SBATCH_PARTITION="${SBATCH_PARTITION:-stampede}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-4}"
SBATCH_TIME="${SBATCH_TIME:-09:00:00}"
SBATCH_JOB_NAME="${SBATCH_JOB_NAME:-craytracer_sweep}"

echo "[submit] project root: ${PROJECT_ROOT}"
echo "[submit] slurm script: ${SLURM_SCRIPT}"
echo "[submit] logs:         ${METRICS_ROOT}/slurm_${SBATCH_JOB_NAME}_<jobid>.{out,err}"

sbatch \
    --partition="${SBATCH_PARTITION}" \
    --cpus-per-task="${SBATCH_CPUS_PER_TASK}" \
    --time="${SBATCH_TIME}" \
    --job-name="${SBATCH_JOB_NAME}" \
    --chdir="${PROJECT_ROOT}" \
    --output="${METRICS_ROOT}/slurm_%x_%j.out" \
    --error="${METRICS_ROOT}/slurm_%x_%j.err" \
    --export=ALL,CRAYTRACER_PROJECT_ROOT="${PROJECT_ROOT}" \
    "${SLURM_SCRIPT}" "$@"
