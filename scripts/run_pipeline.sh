#!/usr/bin/env bash
# Thin wrapper around scripts/run_pipeline.py
# Examples:
# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_pipeline.sh --config configs/csa_mnid_bps.yaml > logs/run_pipeline_csa.log 2>&1 &
# nohup bash scripts/run_pipeline.sh --config configs/lr_v0_nomnid_bps_beats.yaml > logs/run_pipeline_lrv0.log 2>&1 &
# nohup bash scripts/run_pipeline.sh --config configs/lr_v0_mnid_bps_beats.yaml > logs/run_pipeline_lrv0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_pipeline.sh --config configs/lr_v1_nomnid_bps_beats.yaml > logs/run_pipeline_lrv1.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

usage() {
  cat <<EOF
Usage: $0 [--config PATH] [--run-dir PATH] [--method NAME] [--motif-only] [--help]

Defaults:
  --config configs/baseline_bps_csa.yaml (baseline)
  --config configs/lr_v0_nomnid_bps_beats.yaml (if --motif-only and no config provided)

Notes:
  --motif-only skips MNID and uses csv_note_dir from config.
EOF
}

cmd=("${PYTHON_BIN}" "${ROOT}/scripts/run_pipeline.py")

if [[ $# -eq 0 ]]; then
  usage
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config|--run-dir|--method)
      cmd+=("$1" "$2"); shift 2;;
    --motif-only)
      cmd+=("$1"); shift 1;;
    -h|--help)
      usage; exit 0;;
    *)
      cmd+=("$1"); shift 1;;
  esac
done

echo "Running: ${cmd[*]}"
exec "${cmd[@]}"
