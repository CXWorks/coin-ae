"""
eval_open_baseline.py - Vanilla open-source LLM baseline for MUF detection.

Reproduces the open-source comparison reported in the paper (Figure 2):

    COIN (Llama 3.2 3B, fine-tuned)  AUPRC ~ 0.82
    Llama 3.2 3B (vanilla)           AUPRC ~ 0.02
    Llama 3.1 8B (vanilla)           AUPRC ~ 0.02
    Qwen3 4B (vanilla)               AUPRC ~ 0.03

For each test function, the script formats the same prompt as the Coin
classifier ("Is this unsafe? Yes/No"), runs a single forward pass through
the vanilla base model, and extracts the logits of the "Yes" and "No"
tokens at the final position. The unsafe probability is softmax(Yes)/Z,
which is then used for AUPRC, precision, and recall.

Usage:
    python eval_open_baseline.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --data /path/to/coin_test.pkl \
        --n 8000 \
        --output /tmp/baseline_llama32.json

Models reported in the paper (paste any HuggingFace ID):
    --model meta-llama/Llama-3.2-3B-Instruct       # 3B
    --model meta-llama/Llama-3.1-8B-Instruct       # 8B
    --model Qwen/Qwen3-4B-Instruct                 # 4B (uses Qwen3 chat format)
"""
import argparse
import json
import os
import pickle
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support)
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

os.environ.setdefault("WANDB_DISABLED", "true")

PROMPT_TEMPLATE = """Here is a Rust code and please check if the function starting with `>` is safe or unsafe:
    {}

    Is this function unsafe? Answer with "Yes" or "No".

    SOLUTION
    The correct answer is: \""""


def load_test_examples(pkl_path: str, max_n: int = 0, seed: int = 42):
    """Load coin_test.pkl and format (prompt_text, label) pairs.

    Mirrors the safe/unsafe handling used by code/eval_repro.py.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    examples = []
    for entry in data["safe"]:
        f_, text, ls, window = entry
        for idx, (st, ed) in enumerate(ls):
            wst, wed = window[idx]
            lines = text.split("\n")
            for i in range(st - 1, min(ed, len(lines))):
                lines[i] = ">\t" + lines[i]
            snippet = "\n".join(lines[max(0, wst - 1):wed])
            examples.append((PROMPT_TEMPLATE.format(snippet), 0))

    for entry in data["unsafe"]:
        f_, text, ls, window = entry
        for idx, (st, ed) in enumerate(ls):
            wst, wed = window[idx]
            lines = text.split("\n")
            if st - 1 < len(lines):
                lines[st - 1] = lines[st - 1].replace("unsafe ", "", 1)
            for i in range(st - 1, min(ed, len(lines))):
                lines[i] = ">\t" + lines[i]
            snippet = "\n".join(lines[max(0, wst - 1):wed])
            examples.append((PROMPT_TEMPLATE.format(snippet), 1))

    if max_n > 0 and len(examples) > max_n:
        random.seed(seed)
        safe_ex = [e for e in examples if e[1] == 0]
        unsafe_ex = [e for e in examples if e[1] == 1]
        ratio = len(safe_ex) / len(examples)
        n_safe = int(max_n * ratio)
        n_unsafe = max_n - n_safe
        examples = (random.sample(safe_ex, min(n_safe, len(safe_ex)))
                    + random.sample(unsafe_ex, min(n_unsafe, len(unsafe_ex))))
        random.shuffle(examples)
    return examples


def find_yes_no_ids(tokenizer):
    """Get token IDs for 'Yes' and 'No' as they appear after the prompt."""
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    return yes_id, no_id


@torch.no_grad()
def predict_unsafe_prob(model, tokenizer, prompts, yes_id, no_id,
                        max_seq_length, device, batch_size=4):
    """For each prompt, return P(Yes | prompt) over {Yes, No} tokens."""
    enc = [tokenizer(p, truncation=True, max_length=max_seq_length,
                     return_tensors="pt").input_ids[0] for p in prompts]
    order = sorted(range(len(enc)), key=lambda i: enc[i].shape[0])

    probs = [0.0] * len(prompts)
    buf = []
    for k, i in enumerate(tqdm(order, desc="forward", file=sys.stderr)):
        buf.append((i, enc[i]))
        flush = ((k + 1) == len(order)) or (
            len(buf) == batch_size
            or (buf and buf[-1][1].shape[0] != enc[i].shape[0])
        )
        if not flush:
            continue
        ids = torch.stack([t for _, t in buf]).to(device)
        out = model(ids, use_cache=False)
        logits = out.logits[:, -1, :]  # last-token logits
        last2 = torch.stack([logits[:, yes_id], logits[:, no_id]], dim=-1)
        p_yes = torch.softmax(last2, dim=-1)[:, 0]
        for (orig_idx, _), p in zip(buf, p_yes.tolist()):
            probs[orig_idx] = float(p)
        buf = []
    return probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HuggingFace model ID (e.g. meta-llama/Llama-3.2-3B-Instruct)")
    p.add_argument("--data", required=True, help="path to coin_test.pkl")
    p.add_argument("--n", type=int, default=8000,
                   help="stratified sample size (0 = full)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max_seq_length", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--output", required=True)
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--no_4bit", action="store_true",
                   help="disable 4-bit loading (use bf16)")
    args = p.parse_args()

    examples = load_test_examples(args.data, args.n)
    print(f"Loaded {len(examples)} examples "
          f"({sum(1 for _, l in examples if l == 0)} safe, "
          f"{sum(1 for _, l in examples if l == 1)} unsafe)", file=sys.stderr)

    print(f"Loading {args.model} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.no_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto")
    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, device_map="auto")
    model.eval()
    device = next(model.parameters()).device

    yes_id, no_id = find_yes_no_ids(tokenizer)
    print(f"Yes token id: {yes_id}, No token id: {no_id}", file=sys.stderr)

    prompts = [e[0] for e in examples]
    labels = np.array([e[1] for e in examples])

    probs = predict_unsafe_prob(model, tokenizer, prompts, yes_id, no_id,
                                args.max_seq_length, device, args.batch_size)
    probs = np.array(probs)
    preds = (probs > args.threshold).astype(int)

    auprc = float(average_precision_score(labels, probs))
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1])
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    result = {
        "model": args.model, "data": args.data,
        "n": len(examples), "threshold": args.threshold,
        "auprc": auprc, "accuracy": float(accuracy_score(labels, preds)),
        "precision_safe": float(prec[0]),   "recall_safe": float(rec[0]),
        "f1_safe": float(f1[0]),
        "precision_unsafe": float(prec[1]), "recall_unsafe": float(rec[1]),
        "f1_unsafe": float(f1[1]),
        "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
    }

    print(f"\nAUPRC                : {auprc:.4f}")
    print(f"Precision (unsafe)   : {result['precision_unsafe']:.4f}")
    print(f"Recall    (unsafe)   : {result['recall_unsafe']:.4f}")

    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
