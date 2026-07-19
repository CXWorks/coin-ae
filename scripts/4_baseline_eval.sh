#!/bin/bash
# Reproduce the baseline comparisons reported in the paper:
#   (a) Open-source models — vanilla Llama 3.2 3B / 3.1 8B / Qwen3 4B
#       Paper Figure 2 (AUPRC), reproduced with code/eval_open_baseline.py.
#   (b) Closed-source APIs — GPT-4o and Claude-3.7
#       Paper Table 2 (few-shot N=1,3,5) and Table 3 (Best of K=1,3,5),
#       reproduced with code/eval_api_baseline.py.
#
# Usage:
#   bash scripts/4_baseline_eval.sh /path/to/coin_test.pkl  open    # (a) only
#   bash scripts/4_baseline_eval.sh /path/to/coin_test.pkl  api     # (b) only
#   bash scripts/4_baseline_eval.sh /path/to/coin_test.pkl          # both
#
# API mode needs:
#   export OPENAI_API_KEY=...
#   export ANTHROPIC_API_KEY=...
#
# Open-source mode needs HuggingFace access for the gated Llama models:
#   export HF_TOKEN=hf_...   (or `huggingface-cli login`)

set -e

DATA="${1:-data/coin_test.pkl}"
MODE="${2:-both}"
ENV_PATH="${ENV_PATH:-./env}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
N_OPEN="${N_OPEN:-8000}"     # samples for open-source models
N_API="${N_API:-200}"        # samples per API config (cost-controlled)
SHOTS="${SHOTS:-data/shots.jsonl}"

if [ ! -f "$DATA" ]; then
    echo "Error: dataset file not found: $DATA"; exit 1
fi

LD_LIB="$ENV_PATH/lib/python3.11/site-packages/nvidia/cu13/lib"
OUT_DIR="/tmp/coin_baselines"
mkdir -p "$OUT_DIR"

# ----------------------------------------------------------------------
# (a) Open-source baselines
# ----------------------------------------------------------------------
if [ "$MODE" = "open" ] || [ "$MODE" = "both" ]; then
    OPEN_MODELS=(
        "meta-llama/Llama-3.2-3B-Instruct"
        "meta-llama/Llama-3.1-8B-Instruct"
        "Qwen/Qwen3-4B-Instruct"
    )
    for M in "${OPEN_MODELS[@]}"; do
        TAG=$(echo "$M" | tr '/.' '__')
        OUT="$OUT_DIR/open_${TAG}.json"
        echo
        echo "=== Open-source baseline: $M (~$N_OPEN samples) ==="
        CUDA_VISIBLE_DEVICES=$GPU \
        LD_LIBRARY_PATH="$LD_LIB:$LD_LIBRARY_PATH" \
            "$ENV_PATH/bin/python" code/eval_open_baseline.py \
            --model "$M" \
            --data "$DATA" \
            --n "$N_OPEN" \
            --output "$OUT"
    done
fi

# ----------------------------------------------------------------------
# (b) API baselines (GPT-4o / Claude-3.7)
# ----------------------------------------------------------------------
if [ "$MODE" = "api" ] || [ "$MODE" = "both" ]; then
    if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "WARNING: neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set."
        echo "Skipping API baselines."
    else
        # ------------- Table 2: few-shot N=1,3,5 (Best@1) ----------------
        for N_SHOT in 0 1 3 5; do
            if [ -n "$OPENAI_API_KEY" ]; then
                OUT="$OUT_DIR/api_gpt4o_n${N_SHOT}.json"
                echo
                echo "=== GPT-4o, few-shot N=$N_SHOT, Best@1 ==="
                "$ENV_PATH/bin/python" code/eval_api_baseline.py \
                    --provider openai --model gpt-4o \
                    --data "$DATA" --n "$N_API" \
                    --shots_file "$SHOTS" --shots_n "$N_SHOT" \
                    --best_of 1 --output "$OUT"
            fi
            if [ -n "$ANTHROPIC_API_KEY" ]; then
                OUT="$OUT_DIR/api_claude37_n${N_SHOT}.json"
                echo
                echo "=== Claude-3.7, few-shot N=$N_SHOT, Best@1 ==="
                "$ENV_PATH/bin/python" code/eval_api_baseline.py \
                    --provider anthropic --model claude-3-7-sonnet-20250219 \
                    --data "$DATA" --n "$N_API" \
                    --shots_file "$SHOTS" --shots_n "$N_SHOT" \
                    --best_of 1 --output "$OUT"
            fi
        done

        # ------------- Table 3: Best of K=3,5 with N=1 shot --------------
        for K in 3 5; do
            if [ -n "$OPENAI_API_KEY" ]; then
                OUT="$OUT_DIR/api_gpt4o_n1_k${K}.json"
                echo
                echo "=== GPT-4o, N=1, Best of K=$K ==="
                "$ENV_PATH/bin/python" code/eval_api_baseline.py \
                    --provider openai --model gpt-4o \
                    --data "$DATA" --n "$N_API" \
                    --shots_file "$SHOTS" --shots_n 1 \
                    --best_of "$K" --output "$OUT"
            fi
            if [ -n "$ANTHROPIC_API_KEY" ]; then
                OUT="$OUT_DIR/api_claude37_n1_k${K}.json"
                echo
                echo "=== Claude-3.7, N=1, Best of K=$K ==="
                "$ENV_PATH/bin/python" code/eval_api_baseline.py \
                    --provider anthropic --model claude-3-7-sonnet-20250219 \
                    --data "$DATA" --n "$N_API" \
                    --shots_file "$SHOTS" --shots_n 1 \
                    --best_of "$K" --output "$OUT"
            fi
        done
    fi
fi

echo
echo "All baselines complete. JSON results in: $OUT_DIR"
ls -1 "$OUT_DIR" | sort
