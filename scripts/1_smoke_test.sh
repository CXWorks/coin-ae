#!/bin/bash
# Kick-the-tires: classifier evaluation on an 8K stratified sample.
# Expected: AUPRC approx 0.75 in ~20 minutes on one A100-80GB.
#
# Usage: bash scripts/1_smoke_test.sh /path/to/coin_test.pkl

set -e

DATA="${1:-/path/to/coin_test.pkl}"
ENV_PATH="${ENV_PATH:-./env}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
OUT="${OUT:-/tmp/eval_smoke.json}"

if [ ! -f "$DATA" ]; then
    echo "Error: dataset file not found: $DATA"
    echo "Usage: bash scripts/1_smoke_test.sh /path/to/coin_test.pkl"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU \
LD_LIBRARY_PATH="$ENV_PATH/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH" \
    "$ENV_PATH/bin/python" code/eval_repro.py \
    --checkpoint model/llama3.2 \
    --data "$DATA" \
    --n 8000 \
    --output "$OUT"

echo
echo "Smoke test done. Results: $OUT"
echo "Expected: AUPRC ~ 0.75 on 8K stratified sample."
