#!/usr/bin/env bash
# nohup bash scripts/run_slice.sh --method CSA --save-predictions \
#   --out-dir runs/slice_csa_29_21_06 --num-workers 3 > logs/run_slice_csa.log 2>&1 &
# nohup bash scripts/run_slice.sh --method SIATEC --save-predictions \
#   --out-dir runs/slice_siatec_29_21_06 --num-workers 3 > logs/run_slice_siatec.log 2>&1 &
# nohup bash scripts/run_slice.sh --method SIATEC_CS --save-predictions \
#   --out-dir runs/slice_siatec_cs_29_21_06 --num-workers 3 > logs/run_slice_siatec_cs.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

# Defaults
METHOD="SIATEC"
PIECES="21-1,29-1,06-1"
CSV_NOTE_DIR="${ROOT}/runs/baseline_bps_siatec-20251218-173403/predictions"
CSV_LABEL_DIR="${ROOT}/datasets/Beethoven_motif-main/csv_label"
MOTIF_MIDI_DIR="${ROOT}/datasets/Beethoven_motif-main/motif_midi"
OUT_DIR=""
SAVE_PRED=false
NUM_WORKERS=1

usage() {
  cat <<EOF
Usage: $0 [--method CSA|SIATEC|SIATEC_CS] [--pieces CSV] [--csv-note-dir DIR] [--csv-label-dir DIR] [--motif-midi-dir DIR] [--out-dir DIR] [--save-predictions]

Defaults:
  --method          ${METHOD}
  --pieces          ${PIECES}
  --csv-note-dir    ${CSV_NOTE_DIR}
  --csv-label-dir   ${CSV_LABEL_DIR}
  --motif-midi-dir  ${MOTIF_MIDI_DIR}
  --num-workers     ${NUM_WORKERS}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2;;
    --pieces) PIECES="$2"; shift 2;;
    --csv-note-dir) CSV_NOTE_DIR="$2"; shift 2;;
    --csv-label-dir) CSV_LABEL_DIR="$2"; shift 2;;
    --motif-midi-dir) MOTIF_MIDI_DIR="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --save-predictions) SAVE_PRED=true; shift 1;;
    --num-workers) NUM_WORKERS="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1;;
  esac
done

cmd=("${PYTHON_BIN}" "${ROOT}/scripts/run_slice.py"
  --method "${METHOD}"
  --csv_note_dir "${CSV_NOTE_DIR}"
  --csv_label_dir "${CSV_LABEL_DIR}"
  --motif_midi_dir "${MOTIF_MIDI_DIR}"
  --pieces "${PIECES}"
  --num_workers "${NUM_WORKERS}"
)

if [[ -n "${OUT_DIR}" ]]; then
  cmd+=(--output_dir "${OUT_DIR}")
fi
if [[ "${SAVE_PRED}" == "true" ]]; then
  cmd+=(--save_predictions)
fi

echo "Running: ${cmd[*]}"
exec "${cmd[@]}"
