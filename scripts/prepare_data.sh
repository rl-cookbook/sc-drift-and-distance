#!/usr/bin/env bash
# Download GSM8K and DAPO-Math-17k / AIME with SkyRL's own scripts, then build the parquets the runs use.
#   usage: bash scripts/prepare_data.sh [data_dir]     (default: ~/data/drift; run.sh reads DATA_DIR)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HOME/data/drift}
SKYRL_DIR=${SKYRL_DIR:-$HERE/../SkyRL}
cd "$SKYRL_DIR"
uv run --isolated --extra fsdp examples/train/gsm8k/gsm8k_dataset.py --output_dir "$OUT/raw/gsm8k"
DATA_DIR=$OUT/raw/dapo bash examples/train/algorithms/dapo/prepare_dapo_data.sh
uv run --isolated --extra fsdp python "$HERE/prepare_data.py" "$OUT/raw" "$OUT"
