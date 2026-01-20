#!/usr/bin/env bash
set -euo pipefail

python scripts/train_lr_v1.py --config configs/lr_v1_train_bps_pitchbins_beats.yaml
