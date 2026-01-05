#!/usr/bin/env bash
# nohup bash ./scripts/run_baseline.sh --config configs/baseline_bps_csa.yaml --gpu 0 > logs/run_baseline_csa.log 2>&1 &
# nohup bash ./scripts/run_baseline.sh --config configs/baseline_bps_siatec.yaml --gpu 1 > logs/run_baseline_siatec.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT}/configs/baseline_bps.yaml"
RUN_DIR=""
GPU_IDS=""
PYTHON_BIN="${PYTHON:-python3}"

usage() {
  cat <<EOF
Usage: $0 [--config PATH] [--run-dir PATH] [--gpu IDS]

Defaults:
  --config   ${CONFIG}

Notes:
  - Set --gpu (e.g., "0" or "0,1") to export CUDA_VISIBLE_DEVICES before running.
  - To force CPU, set mnid.cpu: true in the config.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"; shift 2;;
    --run-dir)
      RUN_DIR="$2"; shift 2;;
    --gpu)
      GPU_IDS="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1" >&2
      usage; exit 1;;
  esac
done

if [[ -n "${GPU_IDS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

cmd=("${PYTHON_BIN}" "${ROOT}/scripts/run_baseline.py" "--config" "${CONFIG}")
if [[ -n "${RUN_DIR}" ]]; then
  cmd+=("--run-dir" "${RUN_DIR}")
fi

echo "Running: ${cmd[*]}"
exec "${cmd[@]}"
