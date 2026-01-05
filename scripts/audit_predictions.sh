#!/usr/bin/env bash
# nohup bash scripts/audit_predictions.sh \
#   --patterns-dir runs/baseline_bps_csa-20251218-173359/motifs_csa \
#   --out-dir runs/baseline_bps_csa-20251218-173359/analysis_csa > logs/audit_predictions_csa.log 2>&1 &
# nohup bash scripts/audit_predictions.sh \
#   --patterns-dir runs/baseline_bps_siatec-20251218-173403/motifs_siatec \
#   --out-dir runs/baseline_bps_siatec-20251218-173403/analysis_siatec > logs/audit_predictions_siatec.log 2>&1 &
# nohup bash scripts/audit_predictions.sh \
#   --patterns-dir runs/baseline_bps_siatec-20251218-173403/motifs_siatec \
#   --out-dir runs/baseline_bps_siatec-20251218-173403/analysis_siatec_capped_4000 \
#   --num-workers 8 --pairwise-cap 4000 > logs/audit_predictions_siatec.log 2>&1 &
# nohup bash scripts/audit_predictions.sh \
#   --patterns-dir runs/slice_siatec_cs_29_21_06/motifs \
#   --out-dir runs/slice_siatec_cs_29_21_06/analysis \
#   --num-workers 3 --pairwise-cap 4000 > logs/audit_predictions_siatec_cs.log 2>&1 &
# nohup bash scripts/audit_predictions.sh \
#   --patterns-dir runs/slice_siatec_29_21_06/motifs \
#   --out-dir runs/slice_siatec_29_21_06/analysis \
#   --num-workers 3 --pairwise-cap 4000 > logs/audit_predictions_siatec.log 2>&1 &
# nohup bash scripts/audit_predictions.sh \
#   --patterns-dir runs/slice_csa_29_21_06/motifs \
#   --out-dir runs/slice_csa_29_21_06/analysis \
#   --num-workers 3 --pairwise-cap 4000 > logs/audit_predictions_csa.log 2>&1 &


set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LATEST_MOTIFS="$(ls -dt "${ROOT}"/runs/*/motifs_* 2>/dev/null | head -1 || true)"
PATTERNS_DIR="${LATEST_MOTIFS:-${ROOT}/runs/baseline_bps_csa-20251218-161002/motifs_csa}"
OUT_DIR="${PATTERNS_DIR%/motifs_*}/analysis"
NUM_WORKERS=1
MAX_MOTIFS=""
PAIRWISE_CAP=""
PIECES=""
PIECE_PATTERN="*.json"
PYTHON_BIN="${PYTHON:-python3}"

usage() {
  cat <<EOF
Usage: $0 [--patterns-dir DIR] [--out-dir DIR] [--num-workers N] [--max-motifs-per-piece N] [--pairwise-cap N] [--pieces CSV] [--piece-pattern GLOB]

Defaults:
  --patterns-dir ${PATTERNS_DIR}
  --out-dir      ${OUT_DIR}
  --num-workers  ${NUM_WORKERS}
  --piece-pattern ${PIECE_PATTERN}

EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --patterns-dir) PATTERNS_DIR="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --num-workers) NUM_WORKERS="$2"; shift 2;;
    --max-motifs-per-piece) MAX_MOTIFS="$2"; shift 2;;
    --pairwise-cap) PAIRWISE_CAP="$2"; shift 2;;
    --pieces) PIECES="$2"; shift 2;;
    --piece-pattern) PIECE_PATTERN="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1;;
  esac
done

cmd=("${PYTHON_BIN}" "${ROOT}/scripts/audit_predictions.py"
  --patterns_dir "${PATTERNS_DIR}"
  --output_dir "${OUT_DIR}"
  --num_workers "${NUM_WORKERS}"
  --piece_pattern "${PIECE_PATTERN}"
)
if [[ -n "${MAX_MOTIFS}" ]]; then
  cmd+=(--max_motifs_per_piece "${MAX_MOTIFS}")
fi
if [[ -n "${PAIRWISE_CAP}" ]]; then
  cmd+=(--pairwise_cap "${PAIRWISE_CAP}")
fi
if [[ -n "${PIECES}" ]]; then
  cmd+=(--pieces "${PIECES}")
fi

echo "Running: ${cmd[*]}"
exec "${cmd[@]}"
