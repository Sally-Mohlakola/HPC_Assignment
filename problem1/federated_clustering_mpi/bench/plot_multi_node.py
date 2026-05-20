#!/usr/bin/env python3
"""Aggregate the multi-node federated sweep from test_mult_node.sh and emit:

  1) speedup vs num_nodes at fixed processes-per-node  (Strategy A)
  2) speedup vs num_nodes at fixed total processes     (Strategy B)
  3) test accuracy vs round, one curve per (strategy, nodes)
  4) multi_node_summary.csv

Reads one sweep's metrics/sweep_<N>/summary_multinode.csv. Rows are split into
strategies by the recorded (num_nodes, num_processes):
  Strategy A row  iff  num_processes == fixed_ppn   * num_nodes
  Strategy B row  iff  num_processes == fixed_total
A row can satisfy both (e.g. nodes=2, ppn=12 when fixed_total=24, fixed_ppn=12)
and is plotted under both — that's informative, not a bug.

Speedup baseline is the closest-in-time centralised run pooled from every
metrics/sweep_*/summary_centralised.csv, because multi-node sweeps do not have
their own centralised phase.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TS_FMT = "%Y-%m-%d_%H-%M-%S"
RUN_MARKER_RE = re.compile(r"\[mult_node\]\s+run_id=(\S+)")
WORKER_LOG_RE = re.compile(
    r"\[Worker\s+\d+\]\s+Round\s+(\d+)\s+\|\s+Loss:\s+([0-9.eE+-]+)\s+\|\s+"
    r"Train Acc:\s+([0-9.eE+-]+)%"
)
GLOBAL_LOG_RE = re.compile(
    r"\[Central Round\]\s+(\d+)\s+\[Global Test Accuracy\]:\s+([0-9.eE+-]+)%"
)


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s, TS_FMT)


def read_summary(path: Path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def run_dir_name(row):
    """Per-run output directory name: a unique run_id for new data, or the
    start timestamp for legacy data that predates run-id naming."""
    return row.get("run_id") or row["date_time_started"]


def read_run_metrics(path: Path):
    """Returns (epochs, train_loss, train_acc, test_acc) as numpy arrays, or
    None if the file is empty or malformed. A malformed row signals a write
    collision (several runs sharing one timestamped directory), which garbles
    the whole file, so the entire file is rejected rather than partially read."""
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                rows.append((int(r["epoch"]), float(r["train_loss"]),
                             float(r["train_acc"]), float(r["test_acc"])))
            except (TypeError, ValueError, KeyError):
                print(f"[plot] warning: malformed row in {path}; "
                      f"dropping this run's curve", file=sys.stderr)
                return None
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    arr = np.array(rows)
    return arr[:, 0].astype(int), arr[:, 1], arr[:, 2], arr[:, 3]


def run_log_path(sweep_dir: Path, run_id: str):
    direct = sweep_dir / "logs" / f"{run_id}.log"
    if direct.exists():
        return direct

    # Multi-node batch logs group all repetitions for one job.
    m = re.match(r"(.+)_rep\d+_j(\d+)$", run_id)
    if m:
        grouped = sweep_dir / "logs" / f"fed_{m.group(1)}_{m.group(2)}.out"
        if grouped.exists():
            return grouped
    return direct


def read_federated_log_metrics(path: Path, run_id: str):
    active = False
    saw_marker = False
    worker_loss = defaultdict(list)
    worker_acc = defaultdict(list)
    test_acc = {}

    with path.open(errors="replace") as f:
        for line in f:
            marker = RUN_MARKER_RE.search(line)
            if marker:
                marker_run_id = marker.group(1)
                if active and marker_run_id != run_id:
                    break
                active = marker_run_id == run_id
                saw_marker = True
                continue

            if saw_marker and not active:
                continue

            wm = WORKER_LOG_RE.search(line)
            if wm:
                round_id = int(wm.group(1))
                worker_loss[round_id].append(float(wm.group(2)))
                worker_acc[round_id].append(float(wm.group(3)))
                continue

            gm = GLOBAL_LOG_RE.search(line)
            if gm:
                test_acc[int(gm.group(1))] = float(gm.group(2))

    rows = []
    for round_id in sorted(test_acc):
        if not worker_loss[round_id] or not worker_acc[round_id]:
            continue
        rows.append((
            round_id,
            float(np.mean(worker_loss[round_id])),
            float(np.mean(worker_acc[round_id])),
            test_acc[round_id],
        ))
    if not rows:
        return None
    arr = np.array(rows)
    return arr[:, 0].astype(int), arr[:, 1], arr[:, 2], arr[:, 3]


def read_metrics_or_log(csv_path: Path, log_path: Path, run_id: str):
    if csv_path.exists():
        return read_run_metrics(csv_path)
    if log_path.exists():
        curve = read_federated_log_metrics(log_path, run_id)
        if curve is not None:
            print(f"[plot] using log fallback for {run_id}", file=sys.stderr)
            return curve
    print(f"[plot] warning: no metrics found for {csv_path}", file=sys.stderr)
    return None


def avg_curves(curves):
    if not curves:
        return None
    min_len = min(len(c[0]) for c in curves)
    epochs = curves[0][0][:min_len]
    train_loss = np.mean([c[1][:min_len] for c in curves], axis=0)
    train_acc = np.mean([c[2][:min_len] for c in curves], axis=0)
    test_acc = np.mean([c[3][:min_len] for c in curves], axis=0)
    return epochs, train_loss, train_acc, test_acc


def closest_centralised(ts: datetime, centralised_rows):
    best = None
    best_dt = None
    for row in centralised_rows:
        rts = parse_ts(row["date_time_started"])
        delta = abs((rts - ts).total_seconds())
        if best is None or delta < best_dt:
            best = row
            best_dt = delta
    return best


def read_pooled_centralised(metrics_dir: Path):
    rows = []
    for summary in sorted(metrics_dir.glob("sweep_*/summary_centralised.csv")):
        rows.extend(read_summary(summary))
    return rows


def speedup_plot(out_path, title, x_groups, dists, speedups, x_label):
    """x_groups: dict[dist] -> list of (x, mean, std) sorted by x."""
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for dist in dists:
        pts = x_groups.get(dist, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        errs = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=dist)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8,
               label="centralised baseline")
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"Speedup ($t_{centralised} / t_{federated}$)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", required=True, type=Path)
    ap.add_argument("--sweep-id", required=True,
                    help="Read metrics/sweep_<id>/summary_multinode.csv.")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--fixed-ppn", type=int, default=10,
                    help="Strategy A: processes-per-node held constant.")
    ap.add_argument("--fixed-total", type=int, default=24,
                    help="Strategy B: total ranks held constant.")
    ap.add_argument("--dists", type=str, nargs="+",
                    default=["label_shard_noniid"])
    args = ap.parse_args()

    project = args.project_dir
    metrics_dir = project / "metrics"
    sweep_dir = metrics_dir / f"sweep_{args.sweep_id}"
    fdir = sweep_dir / "federated"
    out = sweep_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)

    # Multi-node sweeps have no centralised phase of their own, so speedup is
    # taken against the closest-in-time centralised run from any sweep.
    central_rows = read_pooled_centralised(metrics_dir)
    fed_rows = read_summary(sweep_dir / "summary_multinode.csv")

    if not fed_rows:
        raise SystemExit(f"No multi-node federated runs found in {sweep_dir}")
    if "num_nodes" not in fed_rows[0]:
        raise SystemExit(
            "summary_multinode.csv has no num_nodes column. Rebuild federated and rerun.")

    # ----- Bucket per (strategy, dist, nodes) -------------------------------
    # Each bucket aggregates reps across runs that share the tuple.
    speedups_A = defaultdict(list)  # (dist, nodes) -> [speedup]
    speedups_B = defaultdict(list)
    fed_times_A = defaultdict(list)
    fed_times_B = defaultdict(list)
    np_for_A = {}                   # (dist, nodes) -> num_processes
    np_for_B = {}
    curves_A = defaultdict(list)    # (dist, nodes) -> [curve]
    curves_B = defaultdict(list)
    epochs_to_80_A = defaultdict(list)
    epochs_to_80_B = defaultdict(list)

    for r in fed_rows:
        dist = r["data_distribution"]
        if dist not in args.dists:
            continue
        nodes = int(r["num_nodes"])
        np_count = int(r["num_processes"])
        ts = parse_ts(r["date_time_started"])
        fed_time = float(r["run_time_seconds"])

        in_A = np_count == args.fixed_ppn * nodes
        in_B = np_count == args.fixed_total
        if not (in_A or in_B):
            continue

        match = closest_centralised(ts, central_rows)
        speedup = (float(match["run_time_seconds"]) / fed_time
                   if match is not None else None)

        rid = run_dir_name(r)
        m = fdir / rid / "federated_metrics.csv"
        curve = read_metrics_or_log(m, run_log_path(sweep_dir, rid), rid)

        e80 = (int(r["epochs_to_80"])
               if r.get("epochs_to_80") not in (None, "", "not_reached")
               else None)

        if in_A:
            key = (dist, nodes)
            if speedup is not None:
                speedups_A[key].append(speedup)
            fed_times_A[key].append(fed_time)
            np_for_A[key] = np_count
            if curve is not None:
                curves_A[key].append(curve)
            if e80 is not None:
                epochs_to_80_A[key].append(e80)
        if in_B:
            key = (dist, nodes)
            if speedup is not None:
                speedups_B[key].append(speedup)
            fed_times_B[key].append(fed_time)
            np_for_B[key] = np_count
            if curve is not None:
                curves_B[key].append(curve)
            if e80 is not None:
                epochs_to_80_B[key].append(e80)

    # ----- Plot 1: Strategy A — speedup vs nodes at fixed ppn ---------------
    grouped_A = defaultdict(list)
    for (dist, nodes), vals in speedups_A.items():
        if vals:
            grouped_A[dist].append((nodes, float(np.mean(vals)), float(np.std(vals))))
    for dist in grouped_A:
        grouped_A[dist].sort()
    wrote_A = speedup_plot(
        out / "multinode_speedup_fixed_ppn.png",
        f"Speedup vs number of nodes (fixed ppn={args.fixed_ppn})",
        grouped_A, args.dists, speedups_A,
        x_label="Number of nodes",
    )

    # ----- Plot 2: Strategy B — speedup vs nodes at fixed total -------------
    grouped_B = defaultdict(list)
    for (dist, nodes), vals in speedups_B.items():
        if vals:
            grouped_B[dist].append((nodes, float(np.mean(vals)), float(np.std(vals))))
    for dist in grouped_B:
        grouped_B[dist].sort()
    wrote_B = speedup_plot(
        out / "multinode_speedup_fixed_total.png",
        f"Speedup vs number of nodes (fixed total={args.fixed_total} ranks)",
        grouped_B, args.dists, speedups_B,
        x_label="Number of nodes",
    )

    # ----- Plot 3: test accuracy curves, side-by-side per strategy ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, curves, label_prefix, title in [
        (axes[0], curves_A, "A", f"Test accuracy (fixed ppn={args.fixed_ppn})"),
        (axes[1], curves_B, "B", f"Test accuracy (fixed total={args.fixed_total} ranks)"),
    ]:
        for (dist, nodes) in sorted(curves.keys(), key=lambda k: (k[0], k[1])):
            avg = avg_curves(curves[(dist, nodes)])
            if avg is None:
                continue
            ep, _, _, test_acc = avg
            ax.plot(ep, test_acc, label=f"{dist} N={nodes}")
        ax.axhline(80.0, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Round")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Test accuracy (%)")
    fig.suptitle("Multi-node convergence")
    fig.tight_layout()
    fig.savefig(out / "multinode_test_accuracy.png", dpi=150)
    plt.close(fig)

    # ----- Companion CSV ----------------------------------------------------
    with (out / "multi_node_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "data_distribution", "num_nodes",
                    "num_processes", "speedup_mean", "speedup_std",
                    "fed_time_mean_s", "epochs_to_80_mean", "n_reps"])
        for strategy, sp, ft, npf, e80 in [
            ("fixed_ppn", speedups_A, fed_times_A, np_for_A, epochs_to_80_A),
            ("fixed_total", speedups_B, fed_times_B, np_for_B, epochs_to_80_B),
        ]:
            for (dist, nodes), vals in sorted(sp.items()):
                if not vals:
                    continue
                e80_vals = e80.get((dist, nodes), [])
                w.writerow([
                    strategy, dist, nodes, npf[(dist, nodes)],
                    f"{np.mean(vals):.4f}",
                    f"{np.std(vals):.4f}",
                    f"{np.mean(ft[(dist, nodes)]):.3f}",
                    f"{np.mean(e80_vals):.2f}" if e80_vals else "not_reached",
                    len(vals),
                ])

    missing = []
    if not wrote_A:
        missing.append("Strategy A (no rows matched fixed_ppn × nodes)")
    if not wrote_B:
        missing.append("Strategy B (no rows matched fixed_total)")
    if missing:
        print(f"[plot] skipped: {'; '.join(missing)}")
    print(f"[plot] wrote PNGs and multi_node_summary.csv to {out}")


if __name__ == "__main__":
    main()
