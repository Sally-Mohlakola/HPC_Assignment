#!/usr/bin/env python3
"""Aggregate craytracer benchmark CSV rows and plot sweep results.

Reads metrics/summary.csv, filters to rows written during this benchmark
session (date_time >= --since), averages repeated runs of the same config,
and produces one figure per sweep (block-size, sphere-count, ray-depth)
with three subplots: kernel time, throughput, speedup vs OpenMP.
Each sweep also gets a standalone speedup figure.
"""

import argparse
import csv
import math
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to metrics/summary.csv")
    p.add_argument("--since", required=True,
                   help="Only rows with date_time >= this stamp are used "
                        "(format YYYY-MM-DD_HH-MM-SS)")
    p.add_argument("--output-dir", required=True, help="Where to write plots")
    p.add_argument("--stamp", default="",
                   help="Optional stamp to prefix output filenames")
    p.add_argument("--base-block", type=int, default=256)
    p.add_argument("--base-depth", type=int, default=50)
    p.add_argument("--base-spheres", type=int, default=200,
                   help="Requested sphere count (actual recorded count may "
                        "differ by a small amount due to scene generation)")
    return p.parse_args()


def load_rows(csv_path, since_stamp):
    rows = []
    skipped = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("date_time", "") < since_stamp:
                continue
            for col in INT_COLS:
                row[col] = int(row[col])
            for col in NUMERIC_COLS:
                row[col] = float(row[col])
            if row["implementation"] != "OPENMP" and row["kernel_time_s"] <= 0.0:
                skipped += 1
                continue
            rows.append(row)
    if skipped:
        print(f"  [warn] dropped {skipped} row(s) with non-positive CUDA "
              f"kernel time (failed kernel launches)", file=sys.stderr)
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


def set_zoomed_ylim(ax, values):
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return

    y_min = min(values)
    y_max = max(values)
    if y_min == y_max:
        padding = max(abs(y_min) * 0.05, 1.0)
    else:
        padding = (y_max - y_min) * 0.08

    ax.set_ylim(y_min - padding, y_max + padding)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6))


def plot_sweep(averaged, sweep_var, sweep_label, filter_fn, fixed_desc,
               out_path, plot_title=None):
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

    valid_xs = {r[sweep_var] for impl, rows in by_impl.items()
                if impl != "OPENMP" for r in rows}
    if not valid_xs:
        print(f"  [warn] no CUDA data for sweep over {sweep_var}",
              file=sys.stderr)
        return

    for impl in list(by_impl):
        by_impl[impl] = [r for r in by_impl[impl] if r[sweep_var] in valid_xs]
        if not by_impl[impl]:
            del by_impl[impl]

    for impl in by_impl:
        by_impl[impl].sort(key=lambda r: r[sweep_var])

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    fig.suptitle(plot_title or f"Sweep: {sweep_label}   ({fixed_desc})",
                 fontsize=14)

    ax_time, ax_thr, ax_spd = axes

    impls_present = [i for i in IMPL_ORDER if i in by_impl]
    kernel_times = []
    throughputs = []
    speedups = []
    for impl in impls_present:
        if impl == "OPENMP":
            continue

        rows = by_impl[impl]
        xs = [r[sweep_var] for r in rows]
        style = IMPL_STYLE.get(impl, {})

        times = [r["kernel_time_s"] for r in rows]
        thr = [r["throughput_mpixels_sec"] for r in rows]
        spd = [r["speedup_vs_openmp_kernel"] for r in rows]

        kernel_times.extend(times)
        throughputs.extend(thr)
        speedups.extend(spd)

        ax_time.plot(xs, times, label=impl, **style)
        ax_thr.plot(xs, thr, label=impl, **style)
        ax_spd.plot(xs, spd, label=impl, **style)

    for ax in axes:
        ax.set_xlabel(sweep_label)
        ax.grid(True, alpha=0.3)

    ax_time.set_ylabel("Kernel time (s)")
    ax_time.set_title("Kernel time")
    set_zoomed_ylim(ax_time, kernel_times)

    ax_thr.set_ylabel("Throughput (Mpixels / sec)")
    ax_thr.set_title("Throughput")
    set_zoomed_ylim(ax_thr, throughputs)

    ax_spd.set_ylabel("Speedup vs OpenMP (kernel)")
    ax_spd.set_title("Speedup vs OpenMP")
    set_zoomed_ylim(ax_spd, speedups)

    if sweep_var == "block_size":
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(valid_xs))
            ax.get_xaxis().set_major_formatter(
                matplotlib.ticker.ScalarFormatter())

    handles = []
    labels = []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)),
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_speedup_only(averaged, sweep_var, sweep_label, filter_fn, fixed_desc,
                      out_path):
    """Plot only kernel speedup vs OpenMP for a single sweep."""
    by_impl = defaultdict(list)
    for r in averaged:
        if r["implementation"] == "OPENMP" or not filter_fn(r):
            continue
        by_impl[r["implementation"]].append(r)

    if not by_impl:
        print(f"  [warn] no speedup data for sweep over {sweep_var}",
              file=sys.stderr)
        return

    for impl in by_impl:
        by_impl[impl].sort(key=lambda r: r[sweep_var])

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.set_title(f"Speedup vs OpenMP: {sweep_label} ({fixed_desc})")

    speedups = []
    impls_present = [i for i in IMPL_ORDER if i in by_impl]
    for impl in impls_present:
        rows = by_impl[impl]
        xs = [r[sweep_var] for r in rows]
        spd = [r["speedup_vs_openmp_kernel"] for r in rows]
        style = IMPL_STYLE.get(impl, {})

        speedups.extend(spd)
        ax.plot(xs, spd, label=impl, **style)

    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel(sweep_label)
    ax.set_ylabel("Speedup vs OpenMP (kernel)")
    xticks = sorted({r[sweep_var] for rows in by_impl.values()
                     for r in rows})
    if sweep_var == "block_size":
        ax.set_xscale("log", base=2)
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(xticks)
    ax.grid(True, alpha=0.3)
    set_zoomed_ylim(ax, speedups)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)),
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.tight_layout(rect=(0, 0.15, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.csv, args.since)
    if not rows:
        print(f"No rows in {args.csv} with date_time >= {args.since}",
              file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(rows)} rows (since {args.since})")

    averaged = average_rows(rows)
    run_counts = {r["run_count"] for r in averaged}
    print(f"Averaged into {len(averaged)} (config, implementation) groups; "
          f"runs per group: {sorted(run_counts)}")

    prefix = f"{args.stamp}_" if args.stamp else ""
    avg_csv = out_dir / f"{prefix}averaged.csv"
    write_averaged_csv(averaged, avg_csv)
    print(f"  wrote {avg_csv}")

    base_block = args.base_block
    base_depth = args.base_depth
    base_spheres_actual = closest_num_spheres(averaged, args.base_spheres)
    print(f"Using actual base sphere count = {base_spheres_actual} "
          f"(requested {args.base_spheres})")

    plot_sweep(
        averaged,
        sweep_var="block_size",
        sweep_label="Block Size",
        filter_fn=lambda r: (r["ray_depth"] == base_depth
                             and r["num_spheres"] == base_spheres_actual),
        fixed_desc=f"depth={base_depth}, spheres={base_spheres_actual}",
        out_path=out_dir / f"{prefix}sweep_block_size.png",
        plot_title=(f"Sweep: CUDA block size "
                    f"(depth ={base_depth}, spheres ={base_spheres_actual})"),
    )

    plot_speedup_only(
        averaged,
        sweep_var="block_size",
        sweep_label="Block Size",
        filter_fn=lambda r: (r["ray_depth"] == base_depth
                             and r["num_spheres"] == base_spheres_actual),
        fixed_desc=f"depth={base_depth}, spheres={base_spheres_actual}",
        out_path=out_dir / f"{prefix}sweep_block_size_speedup.png",
    )

    plot_sweep(
        averaged,
        sweep_var="num_spheres",
        sweep_label="Number of spheres",
        filter_fn=lambda r: (r["block_size"] == base_block
                             and r["ray_depth"] == base_depth),
        fixed_desc=f"block={base_block}, depth={base_depth}",
        out_path=out_dir / f"{prefix}sweep_num_spheres.png",
    )

    plot_speedup_only(
        averaged,
        sweep_var="num_spheres",
        sweep_label="Number of spheres",
        filter_fn=lambda r: (r["block_size"] == base_block
                             and r["ray_depth"] == base_depth),
        fixed_desc=f"block={base_block}, depth={base_depth}",
        out_path=out_dir / f"{prefix}sweep_num_spheres_speedup.png",
    )

    plot_sweep(
        averaged,
        sweep_var="ray_depth",
        sweep_label="Max ray bounce depth",
        filter_fn=lambda r: (r["block_size"] == base_block
                             and r["num_spheres"] == base_spheres_actual),
        fixed_desc=f"block={base_block}, spheres={base_spheres_actual}",
        out_path=out_dir / f"{prefix}sweep_ray_depth.png",
    )

    plot_speedup_only(
        averaged,
        sweep_var="ray_depth",
        sweep_label="Max ray bounce depth",
        filter_fn=lambda r: (r["block_size"] == base_block
                             and r["num_spheres"] == base_spheres_actual),
        fixed_desc=f"block={base_block}, spheres={base_spheres_actual}",
        out_path=out_dir / f"{prefix}sweep_ray_depth_speedup.png",
    )


if __name__ == "__main__":
    main()
