#!/usr/bin/env python3
"""Aggregate centralised + federated runs from one test_one_node.sh sweep and
emit the three requested plots:

  1) speedup vs num_processes (per data distribution)
  2) asymptotic test accuracy vs epoch
  3) test accuracy vs training accuracy

Averages over repetitions for each (data_distribution, num_processes).
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
CENTRAL_LOG_RE = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+Loss:\s+([0-9.eE+-]+)\s+\|\s+"
    r"Train Accuracy:\s+([0-9.eE+-]+)%\s+\|\s+"
    r"Test Accuracy:\s+([0-9.eE+-]+)%"
)
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


def read_central_log_metrics(path: Path):
    rows = []
    with path.open(errors="replace") as f:
        for line in f:
            m = CENTRAL_LOG_RE.search(line)
            if m:
                rows.append((
                    int(m.group(1)),
                    float(m.group(2)),
                    float(m.group(3)),
                    float(m.group(4)),
                ))
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    arr = np.array(rows)
    return arr[:, 0].astype(int), arr[:, 1], arr[:, 2], arr[:, 3]


def read_federated_log_metrics(path: Path):
    worker_loss = defaultdict(list)
    worker_acc = defaultdict(list)
    test_acc = {}
    with path.open(errors="replace") as f:
        for line in f:
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


def read_metrics_or_log(csv_path: Path, log_path: Path, log_reader):
    if csv_path.exists():
        return read_run_metrics(csv_path)
    if log_path.exists():
        curve = log_reader(log_path)
        if curve is not None:
            print(f"[plot] using log fallback for {log_path.name}", file=sys.stderr)
            return curve
    print(f"[plot] warning: no metrics found for {csv_path}", file=sys.stderr)
    return None


def avg_curves(curves):
    """curves: list of (epochs, train_loss, train_acc, test_acc).
    Truncates to shortest length so early-stopping reps don't break the mean."""
    if not curves:
        return None
    min_len = min(len(c[0]) for c in curves)
    epochs = curves[0][0][:min_len]
    train_loss = np.mean([c[1][:min_len] for c in curves], axis=0)
    train_acc = np.mean([c[2][:min_len] for c in curves], axis=0)
    test_acc = np.mean([c[3][:min_len] for c in curves], axis=0)
    return epochs, train_loss, train_acc, test_acc


def closest_centralised(ts: datetime, centralised_rows):
    """Pick the centralised summary row with the nearest timestamp."""
    best = None
    best_dt = None
    for row in centralised_rows:
        rts = parse_ts(row["date_time_started"])
        delta = abs((rts - ts).total_seconds())
        if best is None or delta < best_dt:
            best = row
            best_dt = delta
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", required=True, type=Path)
    ap.add_argument("--sweep-id", required=True,
                    help="Read metrics/sweep_<id>/summary_*.csv.")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--nps", type=int, nargs="+", required=True)
    ap.add_argument("--dists", type=str, nargs="+", required=True)
    args = ap.parse_args()

    project = args.project_dir
    sweep_dir = project / "metrics" / f"sweep_{args.sweep_id}"
    cdir = sweep_dir / "centralised"
    fdir = sweep_dir / "federated"
    out = sweep_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)

    central_rows = read_summary(sweep_dir / "summary_centralised.csv")
    fed_rows = read_summary(sweep_dir / "summary_onenode.csv")

    if not central_rows:
        raise SystemExit(f"No centralised runs found in {sweep_dir}")
    if not fed_rows:
        raise SystemExit(f"No one-node federated runs found in {sweep_dir}")

    # ----- Per-run metric curves -------------------------------------------
    central_curves = []
    for r in central_rows:
        m = cdir / run_dir_name(r) / "centralised_metrics.csv"
        log = sweep_dir / "logs" / f"{run_dir_name(r)}.log"
        c = read_metrics_or_log(m, log, read_central_log_metrics)
        if c is not None:
            central_curves.append(c)

    # federated curves bucketed by (dist, np)
    fed_curves = defaultdict(list)
    fed_times = defaultdict(list)
    fed_epochs_to_80 = defaultdict(list)
    speedups = defaultdict(list)

    for r in fed_rows:
        dist = r["data_distribution"]
        np_count = int(r["num_processes"])
        if dist not in args.dists or np_count not in args.nps:
            continue
        ts = parse_ts(r["date_time_started"])
        fed_time = float(r["run_time_seconds"])
        match = closest_centralised(ts, central_rows)
        c_time = float(match["run_time_seconds"])
        speedups[(dist, np_count)].append(c_time / fed_time)
        fed_times[(dist, np_count)].append(fed_time)
        if r["epochs_to_80"] != "not_reached":
            fed_epochs_to_80[(dist, np_count)].append(int(r["epochs_to_80"]))

        m = fdir / run_dir_name(r) / "federated_metrics.csv"
        log = sweep_dir / "logs" / f"{run_dir_name(r)}.log"
        c = read_metrics_or_log(m, log, read_federated_log_metrics)
        if c is not None:
            fed_curves[(dist, np_count)].append(c)

    # ----- Plot 1: speedup vs num processes -------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for dist in args.dists:
        xs, ys, errs = [], [], []
        for np_count in sorted(args.nps):
            vals = speedups.get((dist, np_count), [])
            if not vals:
                continue
            xs.append(np_count)
            ys.append(float(np.mean(vals)))
            errs.append(float(np.std(vals)))
        if xs:
            ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=dist)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8,
               label="centralised baseline")
    ax.set_xlabel("Number of MPI processes (server + workers)")
    ax.set_ylabel(r"Speedup ($t_{centralised} / t_{federated}$)")
    ax.set_title(f"Speedup vs process count (avg of {args.reps} reps)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "speedup_vs_np.png", dpi=150)
    plt.close(fig)

    # ----- Plot 2: asymptotic test accuracy vs epoch ----------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    avg_c = avg_curves(central_curves)
    if avg_c is not None:
        ep, _, _, test_acc = avg_c
        ax.plot(ep, test_acc, label="centralised", linewidth=2, color="black")
    for dist in args.dists:
        for np_count in sorted(args.nps):
            avg = avg_curves(fed_curves.get((dist, np_count), []))
            if avg is None:
                continue
            ep, _, _, test_acc = avg
            ax.plot(ep, test_acc, label=f"{dist} np={np_count}")
    ax.axhline(80.0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Epoch / round")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Asymptotic test accuracy (avg of {args.reps} reps)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "asymptotic_accuracy.png", dpi=150)
    plt.close(fig)

    # ----- Plot 3: test vs training accuracy ------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    if avg_c is not None:
        ep, _, train_acc, test_acc = avg_c
        ax.plot(ep, train_acc, color="black", linestyle="--",
                label="centralised train")
        ax.plot(ep, test_acc, color="black", linestyle="-",
                label="centralised test")
    colour_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ci = 0
    for dist in args.dists:
        for np_count in sorted(args.nps):
            avg = avg_curves(fed_curves.get((dist, np_count), []))
            if avg is None:
                continue
            ep, _, train_acc, test_acc = avg
            colour = colour_cycle[ci % len(colour_cycle)]
            ci += 1
            ax.plot(ep, train_acc, color=colour, linestyle="--",
                    label=f"{dist} np={np_count} train")
            ax.plot(ep, test_acc, color=colour, linestyle="-",
                    label=f"{dist} np={np_count} test")
    ax.set_xlabel("Epoch / round")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Train vs test accuracy (avg of {args.reps} reps)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "train_vs_test.png", dpi=150)
    plt.close(fig)

    # ----- Companion CSV for the report -----------------------------------
    with (out / "speedup_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["data_distribution", "num_processes",
                    "speedup_mean", "speedup_std",
                    "fed_time_mean_s", "epochs_to_80_mean", "n_reps"])
        for dist in args.dists:
            for np_count in sorted(args.nps):
                vals = speedups.get((dist, np_count), [])
                if not vals:
                    continue
                e80 = fed_epochs_to_80.get((dist, np_count), [])
                w.writerow([
                    dist, np_count,
                    f"{np.mean(vals):.4f}",
                    f"{np.std(vals):.4f}",
                    f"{np.mean(fed_times[(dist, np_count)]):.3f}",
                    f"{np.mean(e80):.2f}" if e80 else "not_reached",
                    len(vals),
                ])

    print(f"[plot] wrote PNGs and speedup_summary.csv to {out}")


if __name__ == "__main__":
    main()
