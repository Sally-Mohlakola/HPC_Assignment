#!/bin/bash
set -euo pipefail

# One-node experiment sweep for Problem 1.
#
# Centralised baseline + federated runs across MPI process counts for the
# LABEL_SHARD_NONIID data distribution. Each (configuration, rep) writes a
# timestamped subdir under centralised_metrics/ and federated_metrics/, and
# appends a row to the respective summary.csv. plot_one_node.py then averages
# the 3 reps and renders the requested charts.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

REPS="${REPS:-3}"
NPS=(${NPS:-3 5 7 9 13})

# Stamped before any run so the plotter can ignore older summary rows.
SWEEP_START="$(date +%Y-%m-%d_%H-%M-%S)"

# Single distribution requested for the sweep. Each entry is a name + the
# corresponding -D flags passed to mpicxx via FEDERATED_DEFINES.
DIST_NAMES=("label_shard_noniid")
DIST_DEFINES=("-DLABEL_SHARD_NONIID")

# ---------------------------------------------------------------------------
# Build centralised once (no FEDERATED_DEFINES needed).
# Federated is rebuilt per distribution because the flags are baked in at
# compile time via data_distribution_name() and the #ifdef switches.
# ---------------------------------------------------------------------------
echo "[build] centralised"
make centralised

echo
echo "============================================================"
echo " Centralised baseline: ${REPS} repetitions"
echo "============================================================"
for rep in $(seq 1 "${REPS}"); do
    echo
    echo "[centralised] rep ${rep}/${REPS}"
    (
        cd src
        ./centralised
    )
    # Two distinct timestamps must not collide; cheap guard.
    sleep 1
done

# ---------------------------------------------------------------------------
# Federated sweep: distribution x NP x reps.
# ---------------------------------------------------------------------------
for d_idx in "${!DIST_NAMES[@]}"; do
    dist_name="${DIST_NAMES[$d_idx]}"
    dist_define="${DIST_DEFINES[$d_idx]}"

    echo
    echo "============================================================"
    echo " Federated build: ${dist_name}  (${dist_define})"
    echo "============================================================"
    make clean
    make federated FEDERATED_DEFINES="${dist_define}"

    for np in "${NPS[@]}"; do
        for rep in $(seq 1 "${REPS}"); do
            echo
            echo "[federated] dist=${dist_name} np=${np} rep=${rep}/${REPS}"
            (
                cd src
                mpirun -np "${np}" ./federated
            )
            sleep 1
        done
    done
done

# ---------------------------------------------------------------------------
# Plot. plot_one_node.py reads both summary CSVs and the per-run epoch CSVs,
# averages over reps, and emits PNGs to ./plots/.
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo " Generating plots"
echo "============================================================"
python3 "${PROJECT_DIR}/plot_one_node.py" \
    --project-dir "${PROJECT_DIR}" \
    --reps "${REPS}" \
    --start "${SWEEP_START}" \
    --nps "${NPS[@]}" \
    --dists "${DIST_NAMES[@]}"

echo
echo "[done] plots written to ${PROJECT_DIR}/plots/"
