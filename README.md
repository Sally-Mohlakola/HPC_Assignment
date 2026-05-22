# COMS4040A HPC Assignment

**Student:** Lily de Melo\
**Student Number:** 2545080

**Student:** Gopolang Mohlakola\
**Student Number:** 2680747

## Table of Contents

- [Problem 1 - Federated Clustering with MPI](#problem-1---federated-clustering-with-mpi)
- [Problem 2 - CUDA Ray Tracer](#problem-2---cuda-ray-tracer)

## Problem 1 - Federated Clustering with MPI

This codebase builds and benchmarks MPI-based federated MNIST softmax training. It includes:

- a `centralised` single-process CPU training implementation for baseline timing and accuracy
- a `federated` MPI implementation with one server rank and multiple worker ranks
- compile-time data-distribution options for round-robin and non-IID label-shard experiments
- one-node and multi-node benchmark sweep scripts
- aggregation and plotting scripts for CSV summaries and report figures

### Repository Layout

```text
HPC_Assignment/
└── problem1/
    └── federated_clustering_mpi/
        ├── bench/       # Testing and plotting scripts
        ├── data/        # MNIST train/test data
        ├── metrics/     # Testing logs and plots
        ├── src/         # C++/MPI source code
        ├── makefile   
        └── run.sh       
```

### Setup

Problem 1 needs a C++ compiler and OpenMPI.

From the repository root:

```bash
cd problem1/federated_clustering_mpi
make clean
make all
```

This builds:

- `src/centralised`
- `src/federated`

Available make targets:

```bash
make centralised
make federated
make all
make clean
```

### How To Use It

Build both binaries:

```bash
cd problem1/federated_clustering_mpi
make all
```

Run the binaries directly from `src/`:

```bash
cd src
./centralised
mpirun -np 5 ./federated
```

Run through the helper script:

```bash
cd problem1/federated_clustering_mpi
./run.sh
```

Override the MPI process count:

```bash
NP=9 ./run.sh
```

Build a specific federated data-distribution variant:

```bash
make clean && make federated FEDERATED_DEFINES="-DROUND_ROBIN_BASELINE"
make clean && make federated FEDERATED_DEFINES="-DLABEL_SHARD_NONIID"
make clean && make federated FEDERATED_DEFINES="-DLABEL_SHARD_NONIID -DROTATE_FEATURE_SKEW"
```

If no distribution flag is provided, the federated binary falls back to label-shard non-IID distribution and records it as `label_shard_noniid_default`.

### How To Benchmark It

Run the one-node benchmark sweep:

```bash
cd problem1/federated_clustering_mpi
REPS=3 NPS="3 5 7 9 13 16" CURVE_NPS="3 9 16" bench/test_one_node.sh
```

Run the multi-node Slurm benchmark sweep:

```bash
cd problem1/federated_clustering_mpi
bench/test_mult_node.sh
```

`test_mult_node.sh` submits the Slurm jobs itself. Run the wrapper directly rather than submitting it with `sbatch`.

Each new sweep gets the next integer id from `metrics/.counter` and writes to:

```text
metrics/sweep_<N>/
```

### How To Plot Results

The one-node sweep aggregates and plots automatically. The multi-node sweep aggregates automatically, then prints the plotting command to run locally because the compute nodes may not have `matplotlib`.

Re-aggregate an existing sweep:

```bash
cd problem1/federated_clustering_mpi
bench/aggregate.sh metrics/sweep_<N>
```

Re-plot a one-node sweep:

```bash
python3 bench/plot_one_node.py \
    --project-dir . \
    --sweep-id <N> \
    --reps 3 \
    --nps 3 5 7 9 13 16 \
    --curve-nps 3 9 16 \
    --dists round_robin_iid label_shard_noniid label_shard_noniid_rotate_feature_skew
```

Re-plot a multi-node sweep:

```bash
python3 bench/plot_multi_node.py \
    --project-dir . \
    --sweep-id <N> \
    --reps 3 \
    --fixed-ppn 8 \
    --fixed-total 48
```

### Outputs

#### Single-run outputs

The helper script `./run.sh` increments `metrics/.counter` and writes each single run under:

- `problem1/federated_clustering_mpi/metrics/single_<N>/centralised/centralised/`
- `problem1/federated_clustering_mpi/metrics/single_<N>/federated/federated_np<NP>/`
- `problem1/federated_clustering_mpi/metrics/single_<N>/logs/`

Override the normal run output root with:

```bash
RUN_OUTPUT_DIR=/path/to/output ./run.sh
```

When launched directly from `src/` without the helper script, the binaries default to:

- `problem1/federated_clustering_mpi/centralised_metrics/<run_id>/`
- `problem1/federated_clustering_mpi/federated_metrics/<run_id>/`

Override those direct binary output roots with `CENTRALISED_OUTPUT_ROOT` and `FEDERATED_OUTPUT_ROOT`.

Centralised run directories contain:

- `centralised_metrics.csv`
- `run_summary.csv`

Federated run directories contain:

- `federated_metrics.csv`
- `worker_<rank>_metrics.csv`
- `run_summary.csv`

#### Benchmark outputs

Benchmark sweeps write outputs under `problem1/federated_clustering_mpi/metrics/sweep_<N>/`.

Locations:

- centralised per-run metrics: `metrics/sweep_<N>/centralised/<run_id>/`
- federated per-run metrics: `metrics/sweep_<N>/federated/<run_id>/`
- logs: `metrics/sweep_<N>/logs/`
- plots: `metrics/sweep_<N>/plots/`
- centralised summary, when the sweep includes centralised runs: `metrics/sweep_<N>/summary_centralised.csv`
- one-node federated summary, when the sweep includes one-node federated runs: `metrics/sweep_<N>/summary_onenode.csv`
- multi-node federated summary, when the sweep includes multi-node federated runs: `metrics/sweep_<N>/summary_multinode.csv`

The Python plotting scripts in `bench/` write the following files under
`metrics/sweep_<N>/plots/`.

Generated by `bench/plot_one_node.py`:

- `speedup_vs_np.png`
- `asymptotic_accuracy.png`
- `train_vs_test.png`
- `train_vs_test_np<NP>.png`
- `speedup_summary.csv`

Generated by `bench/plot_multi_node.py`:

- `multinode_speedup_fixed_ppn.png`
- `multinode_speedup_fixed_total.png`
- `multinode_test_accuracy.png`
- `multi_node_summary.csv`

### Experimental Setup

The reported Problem 1 experiments were run on the Wits MS cluster.

#### Hardware

- Slurm partition: `stampede`
- CPU nodes used for MPI runs: Intel Xeon E5-2680-class nodes, with 16 usable CPU cores per node for the benchmark configuration
- Interconnect : Infiniband

#### Software

- C++ compiler: GCC/G++ 11.4.0, building with the C++17 standard
- MPI compiler/runtime: Open MPI 4.1.2 using `/usr/bin/mpicxx` and `/usr/bin/mpirun`
- `mpicxx` backend compiler: `/usr/bin/g++`, GCC 11.4.0 (`Ubuntu 11.4.0-1ubuntu1~22.04.3`)
- Python: Python 3 with `matplotlib` and `numpy` for plotting


### Implementations

- `Centralised`: single-process CPU softmax training baseline
- `Federated`: MPI server/worker training with model aggregation on rank 0
- `Round-robin baseline`: federated data split controlled by `-DROUND_ROBIN_BASELINE`
- `Label-shard non-IID`: federated data split controlled by `-DLABEL_SHARD_NONIID`
- `Rotated feature skew`: optional feature-skew extension controlled by `-DROTATE_FEATURE_SKEW`

### Notes

- Federated runs require at least 3 MPI processes.
- Rank 0 is the server.
- Ranks 1 to `np - 1` are workers/data holders.
- Run `./centralised` and `./federated` from `problem1/federated_clustering_mpi/src`, because the code loads MNIST files through paths relative to that directory.
- The MNIST files must remain under `problem1/federated_clustering_mpi/data`.

## Problem 2 - CUDA Ray Tracer

This codebase builds and benchmarks an OpenMP and CUDA ray tracer. It includes:

- an `OPENMP` CPU renderer for baseline timing
- CUDA renderers using global memory, constant memory, shared memory, and texture memory
- a realistic-scene CUDA variant
- scripts to run benchmark sweeps over block size, object count, and ray depth
- a plotting script to generate averaged CSVs and PNG figures from benchmark outputs

### Repository Layout

```text
HPC_Assignment/
└── problem2/
    └── craytracer/
        ├── bench/          # Testing and plotting scripts
        ├── include/        # Shared ray tracer headers
        ├── metrics/        # Testing logs and plots
        ├── output/         # Rendered images
        ├── renders/        # Main renderer source and CUDA source
        ├── sources/        # Shared C implementation files
        ├── test_textures/  # Texture inputs for CUDA variants
        ├── makefile        
        └── run.sh          
```

### Setup

Problem 2 needs a C compiler with OpenMP support and the CUDA toolkit with `nvcc`.

From the repository root:

```bash
cd problem2/craytracer
make clean
make
```

This builds the ray tracer binary at:

```text
problem2/craytracer/renders/raytracer
```

Build with a specific maximum sphere capacity:

```bash
make clean
make MAX_SPHERES=1024
```

### How To Use It

Build the binary:

```bash
cd problem2/craytracer
make
```

Run the executable directly:

```bash
./renders/raytracer --block-size 256 --max-depth 50 --num-spheres 200
```

Run through the helper script:

```bash
./run.sh --block-size 256 --max-depth 50 --num-spheres 200
```

Runtime options:

```text
--block-size <threads>
--max-depth <ray-bounce-depth>
--num-spheres <sphere-count>
```

Defaults:

- `--block-size 256`
- `--max-depth 50`
- full generated scene if `--num-spheres` is omitted

### How To Benchmark It

Run the full Slurm benchmark sweep:

```bash
cd problem2/craytracer
bench/test.sh
```

`bench/test.sh` is a submission wrapper that calls `sbatch` for `bench/craytracer_sweep.slurm`. Run the wrapper directly rather than submitting it with `sbatch`.

The benchmark sweep runs 3 repeats for:

- CUDA block sizes: `32 64 128 256 512 1024`
- sphere counts: `50 100 200 400 800`
- ray depths: `5 10 25 50 75 100`

Each new sweep gets the next integer id from `metrics/.counter` and writes to:

```text
metrics/sweep_<N>/
```

### How To Plot Results

The benchmark job runs the plotting script automatically after the sweep completes.

Re-plot an existing sweep:

```bash
cd problem2/craytracer
python3 bench/plot_results.py --project-dir . --sweep-id <N>
```

The plotting script reads:

```text
metrics/sweep_<N>/summary.csv
```

and writes averaged data and PNG plots under:

```text
metrics/sweep_<N>/plots/
```

### Outputs

#### Single-run outputs

Single runs write rendered images under `problem2/craytracer/output/`.

Generated image files:

- `openmp.jpg`
- `cuda_global.jpg`
- `cuda_constant.jpg`
- `cuda_shared.jpg`
- `cuda_1d_texture.jpg`
- `cuda_2d_texture.jpg`
- `cuda_2d_texture_constant.jpg`
- `cuda_realistic.jpg`

The helper script `./run.sh` increments `metrics/.counter`, sets `CRAYTRACER_METRICS_DIR`, and writes metrics to:

```text
problem2/craytracer/metrics/single_<N>/summary.csv
problem2/craytracer/metrics/single_<N>/run.log
```

When launched directly without `CRAYTRACER_METRICS_DIR`, the executable appends metrics to:

```text
problem2/craytracer/metrics/summary.csv
```

#### Benchmark outputs

Benchmark sweeps write outputs under `problem2/craytracer/metrics/sweep_<N>/`.

Locations:

- raw per-run logs: `metrics/sweep_<N>/logs/`
- combined metrics CSV: `metrics/sweep_<N>/summary.csv`
- averaged CSV: `metrics/sweep_<N>/plots/averaged.csv`
- PNG plots: `metrics/sweep_<N>/plots/`
- rendered images: `problem2/craytracer/output/`
- Slurm wrapper logs: `problem2/craytracer/metrics/slurm_craytracer_sweep_<jobid>.out` and `.err`

Plot outputs include:

- `sweep_block_size_speedup.png`
- `sweep_num_spheres_speedup.png`

The ray-depth sweep is retained in `summary.csv` and `plots/averaged.csv`; the current plotting script does not emit a separate ray-depth PNG.

### Experimental Setup

The reported Problem 2 experiments were run on the Wits MS cluster.

#### Hardware

- Slurm partition: `stampede`
- GPU: NVIDIA GeForce GTX 1060 6GB
- Global Memory: 6.000 GiB per GPU
- Constant Memory: 64 KiB
- Shared Memory: 48 KiB per block

#### Software

- C compiler: GCC 11.4.0
- CUDA toolkit/compiler: CUDA 12.6, `nvcc` V12.6.68
- CUDA runtime/driver stack: reported by `nvidia-smi`
- Python: Python 3 with `matplotlib` for plotting

#### Double-check commands

Run these on the cluster node or inside the Slurm allocation used for Problem 2:

```bash
scontrol show partition stampede
hostname
lscpu
free -h
gcc --version
nvcc --version
nvidia-smi
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,power.limit,driver_version,compute_cap --format=csv
python3 --version
```

### Implementations

- `OPENMP`: CPU baseline renderer
- `GLOBAL`: CUDA renderer using global-memory scene data
- `CONSTANT`: CUDA renderer using constant memory for scene data
- `SHARED`: CUDA renderer using shared memory for per-block scene data
- `1D TEXTURE`: CUDA renderer using 1D texture memory
- `2D TEXTURE`: CUDA renderer using 2D texture memory
- `2D TEXTURE + CONSTANT`: CUDA renderer combining 2D texture memory and constant memory
- `REALISTIC`: CUDA realistic-scene variant with textured rendering effects

### Notes

- CUDA block size should be positive and should not exceed the device limit, usually 1024 threads per block.
- Requested sphere count should not exceed the compile-time `MAX_SPHERES` capacity.
- The shared-memory CUDA variant stores spheres in block shared memory, so large `MAX_SPHERES` values can exceed the GPU shared-memory limit.
- `run.sh` creates the output directories, rebuilds the binary, and forwards all arguments to `renders/raytracer`.
- `bench/test.sh` must be run on a system where `sbatch` is available.
