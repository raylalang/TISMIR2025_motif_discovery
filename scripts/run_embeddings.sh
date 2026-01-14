#!/usr/bin/env bash

# nohup bash scripts/run_embeddings.sh \
#   --segments-dir runs/segments_bps \
#   --csv-note-dir datasets/Beethoven_motif-main/csv_notes_clean \
#   --out-dir runs/embeddings_bps > logs/run_embeddings.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

SEGMENTS_DIR=""
CSV_NOTE_DIR="${ROOT}/datasets/Beethoven_motif-main/csv_notes_clean"
OUT_DIR=""
PIECES=""
NO_PITCH_CLASS=false

usage() {
  cat <<EOF
Usage: $0 --segments-dir DIR [--csv-note-dir DIR] [--out-dir DIR] [--pieces CSV] [--no-pitch-class]

Defaults:
  --csv-note-dir ${CSV_NOTE_DIR}

Notes:
  - --segments-dir is required (per-piece JSON from run_segments.py).
  - If --out-dir is not set, a timestamped directory under runs/embeddings_<ts> is used.
  - --pieces is comma-separated stems; defaults to all JSONs in segments-dir.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --segments-dir) SEGMENTS_DIR="$2"; shift 2;;
    --csv-note-dir) CSV_NOTE_DIR="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --pieces) PIECES="$2"; shift 2;;
    --no-pitch-class) NO_PITCH_CLASS=true; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1;;
  esac
done

if [[ -z "${SEGMENTS_DIR}" ]]; then
  echo "Error: --segments-dir is required." >&2
  usage
  exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${ROOT}/runs/embeddings_${timestamp}"
fi
mkdir -p "${OUT_DIR}"

cmd=("${PYTHON_BIN}" "${ROOT}/scripts/run_embeddings.py"
  --segments-dir "${SEGMENTS_DIR}"
  --csv-note-dir "${CSV_NOTE_DIR}"
  --output-dir "${OUT_DIR}"
)
if [[ -n "${PIECES}" ]]; then
  cmd+=(--pieces "${PIECES}")
fi
if [[ "${NO_PITCH_CLASS}" == "true" ]]; then
  cmd+=(--no-pitch-class)
fi

echo "Running: ${cmd[*]}"
exec "${cmd[@]}"
