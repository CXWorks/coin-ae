#!/bin/bash
# Evaluate the PoC generator: safe_caller, compiles, ub_detected.
# Optionally re-trains the LoRA adapter from poc_train_v2.jsonl.
#
# Usage:
#   bash scripts/3_poc_eval.sh data/poc_test.jsonl              # eval shipped model
#   TRAIN=1 bash scripts/3_poc_eval.sh data/poc_test.jsonl      # train + eval

set -e

DATA="${1:-data/poc_test.jsonl}"
ENV_PATH="${ENV_PATH:-./env}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_DIR="${MODEL_DIR:-model/llama3.2_poc}"
TRAIN_DATA="${TRAIN_DATA:-data/poc_train_v2.jsonl}"

if [ ! -f "$DATA" ]; then
    echo "Error: test data not found: $DATA"
    echo "Usage: bash scripts/3_poc_eval.sh data/poc_test.jsonl"
    exit 1
fi

LD_LIB="$ENV_PATH/lib/python3.11/site-packages/nvidia/cu13/lib"

if [ "$TRAIN" = "1" ]; then
    echo "Fine-tuning PoC generator from $TRAIN_DATA on GPU $GPU..."
    CUDA_VISIBLE_DEVICES=$GPU \
    LD_LIBRARY_PATH="$LD_LIB:$LD_LIBRARY_PATH" \
        "$ENV_PATH/bin/python" code/train_poc_generator.py
    MODEL_DIR="outputs_poc/final"
fi

echo "Running PoC evaluation with model: $MODEL_DIR"
CUDA_VISIBLE_DEVICES=$GPU \
LD_LIBRARY_PATH="$LD_LIB:$LD_LIBRARY_PATH" \
    "$ENV_PATH/bin/python" code/gen_poc.py --eval \
    --model_dir "$MODEL_DIR" \
    --data "$DATA" \
    --n 20 \
    --output /tmp/poc_eval.json

echo
echo "PoC evaluation done. Results: /tmp/poc_eval.json"
echo "If cargo / miri are not installed, only the safe_caller rate is meaningful."
