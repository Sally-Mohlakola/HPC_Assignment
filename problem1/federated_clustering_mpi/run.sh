#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NP="${NP:-5}"

cd "$PROJECT_DIR"

make all

echo
echo "Running centralised..."
(
    cd src
    ./centralised
)

echo
echo "Running federated with ${NP} MPI processes..."
(
    cd src
    mpirun -np "${NP}" ./federated
)
