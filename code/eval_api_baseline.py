"""
eval_api_baseline.py - GPT-4o / Claude-3.7 baseline driver for MUF classification.

Reproduces the closed-source comparisons in the paper:
  - Table 2 (few-shot N=1,3,5 with Best@1)
  - Table 3 (Best of K=1,3,5 with N=1 shot)

Supports both OpenAI and Anthropic; switch with --provider.

Usage:

    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...

    # Zero-shot GPT-4o, single attempt
    python eval_api_baseline.py \
        --provider openai --model gpt-4o \
        --data /path/to/coin_test.pkl \
        --n 200 \
        --output /tmp/gpt4o_n0.json

    # Few-shot N=3 with Best-of-1, Claude-3.7
    python eval_api_baseline.py \
        --provider anthropic --model claude-3-7-sonnet-20250219 \
        --data /path/to/coin_test.pkl \
        --n 200 \
        --shots_file shots.jsonl --shots_n 3 \
        --output /tmp/claude_n3.json

    # Best-of-K, K=5 (paper Table 3)
    python eval_api_baseline.py \
        --provider openai --model gpt-4o \
        --data /path/to/coin_test.pkl \
        --n 200 --shots_file shots.jsonl --shots_n 1 --best_of 5 \
        --output /tmp/gpt4o_n1_k5.json

The shots.jsonl format (one JSON object per line):
    {"category": "logical memory controls",
     "function_text": "<the prompt-formatted snippet with > markers>",
     "label": 1}

If --shots_file is omitted, the driver runs zero-shot.
"""
import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import Counter

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are an experienced Rust developer. Help me validate whether the "
    "given Rust function is safe or unsafe. Even without `unsafe` operations, "
    "a function can be unsafe because it permits other safe code to trigger "
    "undefined behavior. Reply only `Yes` for unsafe or `No` for safe."
)

USER_TEMPLATE = (
    "The target code and relevant context is below. The target function is "
    "highlighted by `>` at the beginning of the line.\n\n"
    "```rust\n{code}\n```\n\n"
    "Is the target function unsafe?"
)


# ----------------------------------------------------------------------
# Provider abstraction
# ----------------------------------------------------------------------
class OpenAIProvider:
    def __init__(self, model: str):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL",
                                    "https://api.openai.com/v1"),
        )
        self.model = model

    def query(self, system: str, user: str, k: int = 1) -> list[str]:
        out = []
        for _ in range(k):
            for attempt in range(5):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        max_tokens=8, temperature=0.7 if k > 1 else 0.0,
                    )
                    out.append(resp.choices[0].message.content.strip())
                    break
                except Exception as e:
                    print(f"  retry {attempt+1}/5: {e}", file=sys.stderr)
                    time.sleep(2 ** attempt)
            else:
                out.append("")
        return out


class AnthropicProvider:
    def __init__(self, model: str):
        import anthropic
        self.client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
        self.model = model

    def query(self, system: str, user: str, k: int = 1) -> list[str]:
        out = []
        for _ in range(k):
            for attempt in range(5):
                try:
                    resp = self.client.messages.create(
                        model=self.model, max_tokens=8,
                        temperature=0.7 if k > 1 else 0.0,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    out.append(resp.content[0].text.strip())
                    break
                except Exception as e:
                    print(f"  retry {attempt+1}/5: {e}", file=sys.stderr)
                    time.sleep(2 ** attempt)
            else:
                out.append("")
        return out


def build_provider(name: str, model: str):
    if name == "openai":
        return OpenAIProvider(model)
    if name == "anthropic":
        return AnthropicProvider(model)
    raise ValueError(f"unknown provider: {name}")


# ----------------------------------------------------------------------
# Data + prompt assembly
# ----------------------------------------------------------------------
def load_test_examples(pkl_path: str, max_n: int, seed: int):
    """Format (function_text, label) pairs matching the prompt format in eval_repro.py.

    Returns the function text with `>` markers but WITHOUT the
    'Is this unsafe? Answer Yes/No' postfix (that goes in USER_TEMPLATE).
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    examples = []
    for entry in data["safe"]:
        _, text, ls, window = entry
        for idx, (st, ed) in enumerate(ls):
            wst, wed = window[idx]
            lines = text.split("\n")
            for i in range(st - 1, min(ed, len(lines))):
                lines[i] = ">\t" + lines[i]
            examples.append(("\n".join(lines[max(0, wst-1):wed]), 0))

    for entry in data["unsafe"]:
        _, text, ls, window = entry
        for idx, (st, ed) in enumerate(ls):
            wst, wed = window[idx]
            lines = text.split("\n")
            if st - 1 < len(lines):
                lines[st - 1] = lines[st - 1].replace("unsafe ", "", 1)
            for i in range(st - 1, min(ed, len(lines))):
                lines[i] = ">\t" + lines[i]
            examples.append(("\n".join(lines[max(0, wst-1):wed]), 1))

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


def load_shots(path: str, n: int, seed: int):
    """Load few-shot examples; sample N of them (round-robin across categories)."""
    shots = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            shots.append(rec)
    if n <= 0 or not shots:
        return []
    random.seed(seed)
    random.shuffle(shots)
    return shots[:n]


def format_shot(rec: dict) -> str:
    label = "Yes" if rec.get("label", 0) == 1 else "No"
    return (f"Example:\n```rust\n{rec['function_text']}\n```\n"
            f"Is the target function unsafe? Answer: {label}\n")


def build_system_prompt(shots: list) -> str:
    if not shots:
        return SYSTEM_PROMPT
    examples_block = "\n".join(format_shot(s) for s in shots)
    return SYSTEM_PROMPT + "\n\nHere are some labeled examples:\n" + examples_block


def parse_answer(text: str) -> int:
    """Parse Yes/No -> 1/0; ambiguous -> -1."""
    t = text.lower()
    if "yes" in t and "no" not in t:
        return 1
    if "no" in t and "yes" not in t:
        return 0
    return -1


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    p.add_argument("--model", required=True,
                   help="e.g. gpt-4o or claude-3-7-sonnet-20250219")
    p.add_argument("--data", required=True, help="path to coin_test.pkl")
    p.add_argument("--n", type=int, default=200,
                   help="number of test examples (paper uses ~200 due to API cost)")
    p.add_argument("--shots_file", default="",
                   help="path to shots.jsonl for few-shot prompts")
    p.add_argument("--shots_n", type=int, default=0)
    p.add_argument("--best_of", type=int, default=1,
                   help="query K times and treat as unsafe if any returns Yes (Table 3)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set")
    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set")

    examples = load_test_examples(args.data, args.n, args.seed)
    shots = load_shots(args.shots_file, args.shots_n, args.seed) if args.shots_file else []
    system = build_system_prompt(shots)
    print(f"Examples: {len(examples)}  shots: {len(shots)}  best_of: {args.best_of}",
          file=sys.stderr)

    provider = build_provider(args.provider, args.model)
    labels = []
    preds = []
    raw_records = []

    for i, (code, label) in enumerate(tqdm(examples, desc="API")):
        user = USER_TEMPLATE.format(code=code)
        replies = provider.query(system, user, k=args.best_of)
        parsed = [parse_answer(r) for r in replies]
        # Best-of-K: predict unsafe if ANY attempt says unsafe (paper convention)
        if args.best_of > 1:
            pred = 1 if any(p == 1 for p in parsed) else (
                0 if any(p == 0 for p in parsed) else -1)
        else:
            pred = parsed[0]
        labels.append(label); preds.append(pred)
        raw_records.append({"idx": i, "label": label, "replies": replies,
                            "parsed": parsed, "pred": pred})

    labels_a = np.array(labels)
    preds_a  = np.array(preds)
    # Treat ambiguous (-1) as safe (paper convention) to avoid skipping
    preds_eval = np.where(preds_a == -1, 0, preds_a)

    prec, rec, f1, _ = precision_recall_fscore_support(
        labels_a, preds_eval, average=None, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(labels_a, preds_eval, labels=[0, 1])

    result = {
        "provider": args.provider, "model": args.model, "data": args.data,
        "n": len(examples), "shots_n": len(shots), "best_of": args.best_of,
        "accuracy": float(accuracy_score(labels_a, preds_eval)),
        "precision_safe": float(prec[0]),   "recall_safe": float(rec[0]),
        "f1_safe": float(f1[0]),
        "precision_unsafe": float(prec[1]), "recall_unsafe": float(rec[1]),
        "f1_unsafe": float(f1[1]),
        "ambiguous": int((preds_a == -1).sum()),
        "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
    }

    print(f"\nPrecision (unsafe): {result['precision_unsafe']:.4f}")
    print(f"Recall    (unsafe): {result['recall_unsafe']:.4f}")
    print(f"Ambiguous replies: {result['ambiguous']}")

    with open(args.output, "w") as fh:
        json.dump({"summary": result, "raw": raw_records}, fh, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
