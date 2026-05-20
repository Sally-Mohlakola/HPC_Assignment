#!/usr/bin/env python3
"""Aggregate craytracer benchmark CSV rows and plot sweep results.

Reads metrics/sweep_<N>/summary.csv, averages repeated runs of the same
config, and produces one figure per sweep (block-size, sphere-count,
ray-depth) with three subplots: kernel time, throughput, speedup vs OpenMP.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NUMERIC_COLS = [
    "openmp_time_s",
    "kernel_time_s",
    "memory_transfer_time_s",
    "total_without_mem_s",
    "total_with_mem_s",
    "throughput_mpixels_sec",
    "speedup_vs_openmp_kernel",
    "speedup_vs_openmp_with_mem",
]

INT_COLS = ["max_spheres", "ray_depth", "num_spheres", "block_size"]

IMPL_ORDER = [
    "OPENMP",
    "GLOBAL",
    "CONSTANT",
    "SHARED",
    "1D TEXTURE",
    "2D TEXTURE",
    "2D TEXTURE + CONSTANT",
    "REALISTIC",
]

IMPL_STYLE = {
    "OPENMP": {"color": "black", "linestyle": "--", "marker": "x"},
    "GLOBAL": {"color": "#1f77b4", "marker": "o"},
    "CONSTANT": {"color": "#ff7f0e", "marker": "s"},
    "SHARED": {"color": "#2ca02c", "marker": "^"},
    "1D TEXTURE": {"color": "#d62728", "marker": "D"},
    "2D TEXTURE": {"color": "#9467bd", "marker": "v"},
    "2D TEXTURE + CONSTANT": {"color": "#8c564b", "marker": "P"},
    "REALISTIC": {"color": "#e377c2", "marker": "*"},
}

BASE_BLOCK = 256
BASE_DEPTH = 50
BASE_SPHERES = 200


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", required=True, type=Path)
    p.add_argument("--sweep-id", required=True,
                   help="Read metrics/sweep_<id>/summary.csv.")
    return p.parse_args()


def load_rows(csv_path, sweep_id):
    rows = []
    skipped = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("sweep_id") not in (None, "", str(sweep_id)):
                continue
            for col in INT_COLS:
                row[col] = int(row[col])
            for col in NUMERIC_COLS:
                row[col] = float(row[col])
            # A non-positive kernel time means the kernel never actually ran
            # (e.g. a CUDA launch that failed because the block size exceeds
            # the kernel's per-block resource budget). Such rows carry a ~0 s
            # time and a meaningless throughput/speedup, so drop them rather
            # than let them skew the averages and autoscale the plots.
            if row["kernel_time_s"] <= 0.0:
                skipped += 1
                continue
            rows.append(row)
    if skipped:
        print(f"  [warn] dropped {skipped} row(s) with non-positive kernel "
              f"time (failed kernel launches)", file=sys.stderr)
    return rows


def average_rows(rows):
    """Group by (block_size, ray_depth, num_spheres, implementation) and
    average the numeric columns. Returns a list of averaged dicts."""
    groups = defaultdict(list)
    for r in rows:
        key = (r["block_size"], r["ray_depth"], r["num_spheres"],
               r["implementation"])
        groups[key].append(r)

    averaged = []
    for key, group in groups.items():
        block_size, ray_depth, num_spheres, impl = key
        avg = {
            "block_size": block_size,
            "ray_depth": ray_depth,
            "num_spheres": num_spheres,
            "implementation": impl,
            "run_count": len(group),
        }
        for col in NUMERIC_COLS:
            avg[col] = sum(r[col] for r in group) / len(group)
        averaged.append(avg)
    return averaged


def write_averaged_csv(averaged, out_path):
    fieldnames = (["block_size", "ray_depth", "num_spheres", "implementation",
                   "run_count"] + NUMERIC_COLS)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in sorted(averaged, key=lambda r: (
                r["block_size"], r["ray_depth"], r["num_spheres"],
                IMPL_ORDER.index(r["implementation"])
                if r["implementation"] in IMPL_ORDER else 99)):
            w.writerow(row)


def closest_num_spheres(averaged, target):
    """The CSV records actual sphere count, which can drift slightly from
    the requested value. Find the actual count closest to `target` among
    rows in `averaged`."""
    counts = {r["num_spheres"] for r in averaged}
    if not counts:
        return target
    return min(counts, key=lambda c: abs(c - target))


def plot_sweep(averaged, sweep_var, sweep_label, filter_fn, fixed_desc,
               out_path):
    """Plot a sweep: one figure with 3 subplots (kernel time, throughput,
    speedup vs OpenMP). sweep_var is the dict key whose value goes on the
    x-axis. filter_fn(row) selects rows belonging to this sweep."""
    by_impl = defaultdict(list)
    for r in averaged:
        if not filter_fn(r):
            continue
        by_impl[r["implementation"]].append(r)

    if not by_impl:
        print(f"  [warn] no data for sweep over {sweep_var}", file=sys.stderr)
        return

    for impl in by_impl:
        by_impl[impl].sort(key=lambda r: r[sweep_var])

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    fig.suptitle(f"Sweep: {sweep_label}   ({fixed_desc})", fontsize=14)

    ax_time, ax_thr, ax_spd = axes

    impls_present = [i for i in IMPL_ORDER if i in by_impl]
    for impl in impls_present:
        rows = by_impl[impl]
        xs = [r[sweep_var] for r in rows]
        style = IMPL_STYLE.get(impl, {})

        ax_time.plot(xs, [r["kernel_time_s"] for r in rows],
                     label=impl, **style)
        ax_thr.plot(xs, [r["throughput_mpixels_sec"] for r in rows],
                    label=impl, **style)
        if impl != "OPENMP":
            ax_spd.plot(xs, [r["speedup_vs_openmp_kernel"] for r in rows],
                        label=impl, **style)

    for ax in axes:
        ax.set_xlabel(sweep_label)
        ax.grid(True, alpha=0.3)

    ax_time.set_ylabel("Kernel time (s)")
    ax_time.set_title("Kernel time (lower is better)")
    ax_time.set_yscale("log")

    ax_thr.set_ylabel("Throughput (Mpixels / sec)")
    ax_thr.set_title("Throughput (higher is better)")

    ax_spd.set_ylabel("Speedup vs OpenMP (kernel)")
    ax_spd.set_title("Speedup vs OpenMP (higher is better)")
    ax_spd.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")

    if sweep_var == "block_size":
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted({r["block_size"] for r in averaged
                                  if filter_fn(r)}))
            ax.get_xaxis().set_major_formatter(
                matplotlib.ticker.ScalarFormatter())

    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)),
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    args = parse_args()
    sweep_dir = args.project_dir / "metrics" / f"sweep_{args.sweep_id}"
    csv_path = sweep_dir / "summary.csv"
    out_dir = sweep_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path, args.sweep_id)
    if not rows:
        print(f"No rows found in {csv_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(rows)} rows from {csv_path}")

    averaged = average_rows(rows)
    run_counts = {r["run_count"] for r in averaged}
    print(f"Averaged into {len(averaged)} (config, implementation) groups; "
          f"runs per group: {sorted(run_counts)}")

    avg_csv = out_dir / "averaged.csv"
    write_averaged_csv(averaged, avg_csv)
    print(f"  wrote {avg_csv}")

    base_block = BASE_BLOCK
    base_depth = BASE_DEPTH
    base_spheres_actual = closest_num_spheres(averaged, BASE_SPHERES)
    print(f"Using actual base sphere count = {base_spheres_actual} "
          f"(requested {BASE_SPHERES})")

    # A block size is only a meaningful sweep point if at least one CUDA
    # implementation produced a valid run at it. A size where every GPU
    # kernel launch failed (e.g. 1024 exceeds the kernel's per-block resource
    # budget) leaves only the block-size-independent OpenMP baseline, which
    # would just dangle an empty tick on the axis.
    def block_sweep_base(r):
        return (r["ray_depth"] == base_depth
                and r["num_spheres"] == base_spheres_actual)

    gpu_block_sizes = {r["block_size"] for r in averaged
                       if block_sweep_base(r) and r["implementation"] != "OPENMP"}

    plot_sweep(
        averaged,
        sweep_var="block_size",
        sweep_label="CUDA block size (threads)",
        filter_fn=lambda r: (block_sweep_base(r)
                             and r["block_size"] in gpu_block_sizes),
        fixed_desc=f"depth={base_depth}, spheres={base_spheres_actual}",
        out_path=out_dir / "sweep_block_size.png",
    )

    plot_sweep(
        averaged,
        sweep_var="num_spheres",
        sweep_label="Number of spheres",
        filter_fn=lambda r: (r["block_size"] == base_block
                             and r["ray_depth"] == base_depth),
        fixed_desc=f"block={base_block}, depth={base_depth}",
        out_path=out_dir / "sweep_num_spheres.png",
    )

    plot_sweep(
        averaged,
        sweep_var="ray_depth",
        sweep_label="Max ray bounce depth",
        filter_fn=lambda r: (r["block_size"] == base_block
                             and r["num_spheres"] == base_spheres_actual),
        fixed_desc=f"block={base_block}, spheres={base_spheres_actual}",
        out_path=out_dir / "sweep_ray_depth.png",
    )


if __name__ == "__main__":
    main()
