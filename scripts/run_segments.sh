#!/usr/bin/env bash
# nohup bash scripts/run_segments.sh \
#   --csv-note-dir datasets/Beethoven_motif-main/csv_notes_clean \
#   --out-dir runs/segments_bps > logs/run_segments.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

CSV_NOTE_DIR="${ROOT}/datasets/Beethoven_motif-main/csv_notes_clean"
OUT_DIR=""
PIECES=""
SCALES="2.0,4.0,8.0,16.0"
HOP_RATIO="0.25"
MIN_NOTES="3"

usage() {
  cat <<EOF
Usage: $0 [--csv-note-dir DIR] [--out-dir DIR] [--pieces CSV] [--scales LIST] [--hop-ratio R] [--min-notes N]

Defaults:
  --csv-note-dir ${CSV_NOTE_DIR}
  --scales       ${SCALES}
  --hop-ratio    ${HOP_RATIO}
  --min-notes    ${MIN_NOTES}

Notes:
  - If --out-dir is not set, a timestamped directory under runs/segments_<ts> is used.
  - --pieces is a comma-separated list of stems (e.g., 21-1,29-1). Defaults to all CSVs in csv-note-dir.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv-note-dir) CSV_NOTE_DIR="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --pieces) PIECES="$2"; shift 2;;
    --scales) SCALES="$2"; shift 2;;
    --hop-ratio) HOP_RATIO="$2"; shift 2;;
    --min-notes) MIN_NOTES="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1;;
  esac
done

timestamp="$(date +%Y%m%d-%H%M%S)"
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${ROOT}/runs/segments_${timestamp}"
fi
mkdir -p "${OUT_DIR}"

cmd=("${PYTHON_BIN}" "${ROOT}/scripts/run_segments.py"
  --csv-note-dir "${CSV_NOTE_DIR}"
  --output-dir "${OUT_DIR}"
  --scales "${SCALES}"
  --hop-ratio "${HOP_RATIO}"
  --min-notes "${MIN_NOTES}"
)
if [[ -n "${PIECES}" ]]; then
  cmd+=(--pieces "${PIECES}")
fi

echo "Running: ${cmd[*]}"
exec "${cmd[@]}"
