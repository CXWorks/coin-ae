#!/bin/bash
# Full classifier evaluation on all 319,652 test samples.
# Uses 4-GPU sharding (~2.5 hours) or single-GPU fallback (~10 hours).
#
# Usage:
#   bash scripts/2_full_eval.sh /path/to/coin_test.pkl          # 4-GPU
#   NUM_SHARDS=1 bash scripts/2_full_eval.sh /path/to/...        # 1-GPU

set -e

DATA="${1:-/path/to/coin_test.pkl}"
ENV_PATH="${ENV_PATH:-./env}"
NUM_SHARDS="${NUM_SHARDS:-4}"
GPU_BASE="${GPU_BASE:-0}"   # first GPU index; uses GPU_BASE..GPU_BASE+NUM_SHARDS-1

if [ ! -f "$DATA" ]; then
    echo "Error: dataset file not found: $DATA"
    exit 1
fi

LD_LIB="$ENV_PATH/lib/python3.11/site-packages/nvidia/cu13/lib"

if [ "$NUM_SHARDS" = "1" ]; then
    echo "Running single-GPU full eval on GPU $GPU_BASE..."
    CUDA_VISIBLE_DEVICES=$GPU_BASE \
    LD_LIBRARY_PATH="$LD_LIB:$LD_LIBRARY_PATH" \
        "$ENV_PATH/bin/python" code/eval_repro.py \
        --checkpoint model/llama3.2 \
        --data "$DATA" \
        --n 0 \
        --output /tmp/eval_full.json
    echo "Done. Results: /tmp/eval_full.json"
    exit 0
fi

echo "Running $NUM_SHARDS-GPU sharded eval (GPUs $GPU_BASE..$((GPU_BASE+NUM_SHARDS-1)))..."
for SHARD in $(seq 0 $((NUM_SHARDS-1))); do
    GPU=$((GPU_BASE + SHARD))
    echo "  Launching shard $SHARD on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU \
    LD_LIBRARY_PATH="$LD_LIB:$LD_LIBRARY_PATH" \
        "$ENV_PATH/bin/python" code/eval_repro.py \
        --checkpoint model/llama3.2 \
        --data "$DATA" \
        --n 0 --shard_id $SHARD --num_shards $NUM_SHARDS \
        --output /tmp/eval_shard${SHARD}.json &
done
wait

echo
echo "All shards done. Merging results..."
"$ENV_PATH/bin/python" - <<EOF
import pickle, numpy as np, json
from sklearn.metrics import (average_precision_score,
                              precision_recall_fscore_support,
                              precision_recall_curve)

NUM = $NUM_SHARDS
all_probs, all_labels = [], []
for i in range(NUM):
    with open(f"data_shard{i}.pkl", "rb") as fh:
        probs, labels = pickle.load(fh)
    all_probs.extend(probs)
    all_labels.extend(labels)

probs_unsafe = np.array([float(p[1]) for p in all_probs])
labels = np.array(all_labels)

auprc = average_precision_score(labels, probs_unsafe)
preds = (probs_unsafe > 0.7).astype(int)
prec, rec, f1, _ = precision_recall_fscore_support(
    labels, preds, average=None, labels=[0, 1])

p_curve, r_curve, t_curve = precision_recall_curve(labels, probs_unsafe)
mask = r_curve >= 0.80
prec_at_80_recall = p_curve[mask].max() if mask.any() else 0.0

result = {
    "n": len(labels), "auprc": float(auprc),
    "threshold_0.7": {
        "precision_safe": float(prec[0]),  "recall_safe": float(rec[0]),
        "precision_unsafe": float(prec[1]), "recall_unsafe": float(rec[1]),
        "f1_safe": float(f1[0]), "f1_unsafe": float(f1[1]),
    },
    "precision_at_recall_0.80": float(prec_at_80_recall),
}

print(f"AUPRC            : {auprc:.4f}")
print(f"Recall (unsafe)  : {rec[1]:.4f}")
print(f"Precision (unsafe) @ t=0.7  : {prec[1]:.4f}")
print(f"Precision (unsafe) @ rec=0.80: {prec_at_80_recall:.4f}")

with open("/tmp/eval_full_merged.json", "w") as fh:
    json.dump(result, fh, indent=2)
print("\nSaved merged results: /tmp/eval_full_merged.json")
EOF
