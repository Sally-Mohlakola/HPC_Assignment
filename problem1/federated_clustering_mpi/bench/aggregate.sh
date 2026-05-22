#!/bin/bash
set -euo pipefail

# Aggregate one Problem 1 sweep.

SWEEP_DIR="${1:?usage: aggregate.sh <sweep_dir>}"
if [[ ! -d "$SWEEP_DIR" ]]; then
    echo "[aggregate] sweep directory '$SWEEP_DIR' does not exist" >&2
    exit 1
fi

# field_count <csv-line> -> number of comma-separated fields
field_count() { awk -F, '{print NF}' <<<"$1"; }

# collect_rows <kind_dir> <tmp_out> -> shared header
collect_rows() {
    local kind_dir="$1" tmp_out="$2"
    local header="" rs h row
    for rs in "$kind_dir"/*/run_summary.csv; do
        [[ -e "$rs" ]] || continue
        h="$(sed -n '1p' "$rs")"
        row="$(sed -n '2p' "$rs")"
        if [[ -z "$h" || -z "$row" ]]; then
            echo "[aggregate] WARN: ${rs} empty or header-only, skipping" >&2
            continue
        fi
        [[ -z "$header" ]] && header="$h"
        if [[ "$h" != "$header" ]]; then
            echo "[aggregate] WARN: ${rs} header differs from the rest, skipping" >&2
            continue
        fi
        if [[ "$(field_count "$row")" != "$(field_count "$header")" ]]; then
            echo "[aggregate] WARN: ${rs} row field-count mismatch, skipping" >&2
            continue
        fi
        printf '%s\n' "$row" >> "$tmp_out"
    done
    printf '%s' "$header"
}

# Centralised rows.
ctmp="$(mktemp)"
cheader="$(collect_rows "$SWEEP_DIR/centralised" "$ctmp")"
if [[ -n "$cheader" ]]; then
    { printf '%s\n' "$cheader"; sort "$ctmp"; } > "$SWEEP_DIR/summary_centralised.csv"
    echo "[aggregate] centralised: $(wc -l < "$ctmp") run(s) -> summary_centralised.csv"
else
    echo "[aggregate] centralised: no runs found"
fi
rm -f "$ctmp"

# Federated rows split by num_nodes.
ftmp="$(mktemp)"
fheader="$(collect_rows "$SWEEP_DIR/federated" "$ftmp")"
if [[ -n "$fheader" ]]; then
    one="$(mktemp)"; multi="$(mktemp)"
    while IFS= read -r row; do
        [[ -z "$row" ]] && continue
        if [[ "$(awk -F, '{print $5}' <<<"$row")" == "1" ]]; then
            printf '%s\n' "$row" >> "$one"
        else
            printf '%s\n' "$row" >> "$multi"
        fi
    done < "$ftmp"
    if [[ -s "$one" ]]; then
        { printf '%s\n' "$fheader"; sort "$one"; } > "$SWEEP_DIR/summary_onenode.csv"
        echo "[aggregate] federated one-node:   $(wc -l < "$one") run(s) -> summary_onenode.csv"
    fi
    if [[ -s "$multi" ]]; then
        { printf '%s\n' "$fheader"; sort "$multi"; } > "$SWEEP_DIR/summary_multinode.csv"
        echo "[aggregate] federated multi-node: $(wc -l < "$multi") run(s) -> summary_multinode.csv"
    fi
    rm -f "$one" "$multi"
else
    echo "[aggregate] federated: no runs found"
fi
rm -f "$ftmp"
