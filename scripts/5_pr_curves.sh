#!/bin/bash
# Run Coin + every vanilla open-source baseline on the same test sample,
# then draw a single precision-recall figure overlaying all curves
# (reproducing paper Figure 2).
#
# Usage:
#   bash scripts/5_pr_curves.sh /path/to/coin_test.pkl
#
# Environment overrides:
#   N=8000           # number of stratified samples per model
#   GPU=0            # CUDA device index
#   OUT_DIR=/tmp/coin_pr_curves
#   SKIP=llama-3.1   # comma-separated substrings to skip (matched against model id)
#
# Output:
#   <OUT_DIR>/<model>.json + <model>_probs.pkl    per model
#   <OUT_DIR>/pr_curves.png + pr_curves.pdf       overlay figure
#   <OUT_DIR>/summary.txt                         AUPRC table

set -e

DATA="${1:-/path/to/coin_test.pkl}"
ENV_PATH="${ENV_PATH:-./env}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
N="${N:-8000}"
OUT_DIR="${OUT_DIR:-/tmp/coin_pr_curves}"
SKIP="${SKIP:-}"

if [ ! -f "$DATA" ]; then
    echo "Error: dataset file not found: $DATA"
    echo "Usage: bash scripts/5_pr_curves.sh /path/to/coin_test.pkl"
    exit 1
fi

mkdir -p "$OUT_DIR"
LD_LIB="$ENV_PATH/lib/python3.11/site-packages/nvidia/cu13/lib"
PY="$ENV_PATH/bin/python"

skip_match() {
    local id="$1"
    [ -z "$SKIP" ] && return 1
    IFS=',' read -ra patterns <<< "$SKIP"
    for pat in "${patterns[@]}"; do
        [[ "$id" == *"$pat"* ]] && return 0
    done
    return 1
}

run_open_baseline() {
    local label="$1"; local hf_id="$2"; local out_tag="$3"
    if skip_match "$hf_id"; then
        echo "  [skip] $label  ($hf_id)"; return
    fi
    local OUT="$OUT_DIR/${out_tag}.json"
    if [ -s "${OUT%.json}_probs.pkl" ]; then
        echo "  [cached] $label  -> ${OUT%.json}_probs.pkl"; return
    fi
    echo
    echo "=== $label  ($hf_id) ==="
    CUDA_VISIBLE_DEVICES=$GPU \
    LD_LIBRARY_PATH="$LD_LIB:$LD_LIBRARY_PATH" \
        "$PY" code/eval_open_baseline.py \
        --model "$hf_id" \
        --data "$DATA" \
        --n "$N" \
        --output "$OUT"
}

# ---------- (1) Coin fine-tuned classifier ----------
COIN_OUT="$OUT_DIR/coin.json"
COIN_PROBS="$OUT_DIR/coin_probs.pkl"
if [ ! -s "$COIN_PROBS" ]; then
    echo "=== Coin (Llama 3.2 3B, fine-tuned) ==="
    CUDA_VISIBLE_DEVICES=$GPU \
    LD_LIBRARY_PATH="$LD_LIB:$LD_LIBRARY_PATH" \
        "$PY" code/eval_repro.py \
        --checkpoint model/llama3.2 \
        --data "$DATA" \
        --n "$N" \
        --output "$COIN_OUT"
    # eval_repro.py writes the sidecar to ./data.pkl — move it into OUT_DIR
    mv -f data.pkl "$COIN_PROBS"
else
    echo "[cached] Coin -> $COIN_PROBS"
fi

# ---------- (2) Open-source baselines ----------
run_open_baseline "Llama 3.2 3B" "meta-llama/Llama-3.2-3B-Instruct"  open_llama32_3b
run_open_baseline "Llama 3.1 8B" "meta-llama/Llama-3.1-8B-Instruct"  open_llama31_8b
run_open_baseline "Qwen3 4B"     "Qwen/Qwen3-4B-Instruct"            open_qwen3_4b

# ---------- (3) Plot ----------
INPUTS=()
INPUTS+=("Coin:$COIN_PROBS")
for tag in open_llama32_3b open_llama31_8b open_qwen3_4b; do
    PKL="$OUT_DIR/${tag}_probs.pkl"
    if [ -s "$PKL" ]; then
        case "$tag" in
            open_llama32_3b) LABEL="Llama 3.2 3B" ;;
            open_llama31_8b) LABEL="Llama 3.1 8B" ;;
            open_qwen3_4b)   LABEL="Qwen3 4B" ;;
        esac
        INPUTS+=("$LABEL:$PKL")
    fi
done

echo
echo "=== Drawing precision-recall figure ==="
"$PY" code/draw_pr_curves.py \
    --inputs "${INPUTS[@]}" \
    --output "$OUT_DIR/pr_curves.png" \
    --title "Precision-Recall on MUF detection (n=$N)" \
    | tee "$OUT_DIR/summary.txt"

echo
echo "Done. Outputs:"
echo "  $OUT_DIR/pr_curves.png"
echo "  $OUT_DIR/pr_curves.pdf"
echo "  $OUT_DIR/summary.txt"
