"""
infer_batch.py - Run the Coin classifier on a new Rust crate (reusability).

Scans a crate directory for .rs files, extracts every function that is
NOT declared `unsafe fn` (the MUF candidates), builds the same windowed
prompt used during training, and scores each candidate with the
fine-tuned classifier. Results are written to a CSV sorted by predicted
MUF probability (highest first).

Usage:
    python infer_batch.py \
        --crate_dir /path/to/rust/crate \
        --model_dir ../model/llama3.2 \
        --output muf_predictions.csv

Output CSV columns:
    file, start_line, end_line, prob_muf, flagged

Note: candidate extraction here is a lightweight brace-matching scanner
so the tool runs without a Rust toolchain. The paper's data pipeline
uses the rustc 1.83 patch in ../custom_rustc_patch/ for precise
compiler-based extraction; for a quick reusability check the scanner
covers ordinary `fn` items well.
"""

# eval_repro applies the unsloth/peft monkey-patches needed to load the
# checkpoint with its 2-class lm_head; importing it activates them.
from eval_repro import load_model
from unsloth import FastLanguageModel

import argparse
import csv
import os
import re
import sys

import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

PROMPT = """Here is a Rust code and please check if the function starting with `>` is safe or unsafe:
    {}

    Is this function unsafe? Answer with "Yes" or "No".

    SOLUTION
    The correct answer is: \""""

SKIP_DIRS = {"target", "tests", "examples", "benches", ".git"}

# A line that begins a function item, with optional visibility/qualifiers.
FN_RE = re.compile(
    r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?"
    r"(?:default\s+)?(?:const\s+)?(?:async\s+)?"
    r"(?P<unsafe>unsafe\s+)?"
    r"(?:extern\s+\"[^\"]*\"\s+)?"
    r"fn\s+[A-Za-z_]"
)


def find_functions(lines):
    """Yield (start_line, end_line, is_unsafe) for fn items, 1-indexed
    inclusive, via brace matching from the fn signature."""
    i = 0
    n = len(lines)
    while i < n:
        m = FN_RE.match(lines[i])
        if not m:
            i += 1
            continue
        start = i
        depth = 0
        opened = False
        j = i
        while j < n:
            stripped = lines[j]
            # Trait/extern declarations without a body end with ';'
            if not opened and ";" in stripped.split("{")[0]:
                break
            for ch in stripped:
                if ch == "{":
                    depth += 1
                    opened = True
                elif ch == "}":
                    depth -= 1
            if opened and depth <= 0:
                yield start + 1, j + 1, bool(m.group("unsafe"))
                break
            j += 1
            if j - start > 4000:  # unmatched braces safety valve
                break
        i = j + 1 if opened else i + 1


def collect_candidates(crate_dir):
    """Scan crate_dir for safe-declared fn items in non-test .rs files."""
    candidates = []  # (file_path, file_lines, st, ed)
    for root, dirs, files in os.walk(crate_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in files:
            if not fname.endswith(".rs"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", errors="replace") as fh:
                    lines = fh.readlines()
            except OSError:
                continue
            for st, ed, is_unsafe in find_functions(lines):
                if not is_unsafe:
                    candidates.append((fpath, lines, st, ed))
    return candidates


def expand_window(tokenizer, lines, st, ed, max_tokens):
    """Symmetric context expansion via binary search (same scheme used to
    precompute the `window` field of coin_test.pkl)."""
    n = len(lines)
    max_radius = max(st - 1, n - ed)

    def prompt_for(wst, wed):
        marked = list(lines)
        for i in range(st - 1, min(ed, n)):
            marked[i] = ">\t" + marked[i]
        return PROMPT.format("".join(marked[wst - 1:wed]))

    full = prompt_for(1, n)
    if len(tokenizer(full)["input_ids"]) <= max_tokens:
        return full

    lo, hi = 0, max_radius
    best_w = (st, ed)
    while lo <= hi:
        mid = (lo + hi) // 2
        wst = max(1, st - mid)
        wed = min(n, ed + mid)
        text = prompt_for(wst, wed)
        if len(tokenizer(text)["input_ids"]) <= max_tokens:
            best_w = (wst, wed)
            lo = mid + 1
        else:
            hi = mid - 1
    return prompt_for(*best_w)


def main():
    p = argparse.ArgumentParser(
        description="Score every safe-declared function in a Rust crate "
                    "with the Coin MUF classifier.")
    p.add_argument("--crate_dir", required=True,
                   help="path to the Rust crate (or any directory of .rs files)")
    p.add_argument("--model_dir", default="../model/llama3.2",
                   help="fine-tuned classifier checkpoint (LoRA adapter dir)")
    p.add_argument("--output", default="muf_predictions.csv")
    p.add_argument("--threshold", type=float, default=0.8,
                   help="probability above which a function is flagged as MUF")
    p.add_argument("--max_seq_length", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    if not os.path.isdir(args.crate_dir):
        sys.exit(f"Error: --crate_dir not found: {args.crate_dir}")

    print(f"Scanning {args.crate_dir} for candidate functions ...")
    candidates = collect_candidates(args.crate_dir)
    print(f"  {len(candidates)} safe-declared functions found")
    if not candidates:
        sys.exit("No candidates found; nothing to do.")

    print(f"Loading classifier from {args.model_dir} ...")
    model, tokenizer = load_model(args.model_dir)
    FastLanguageModel.for_inference(model)
    model.eval()

    # Budget below max_seq_length so the prompt never gets truncated
    # mid-function.
    budget = args.max_seq_length - 32
    tokenized = []
    for fpath, lines, st, ed in tqdm(candidates, desc="tokenize"):
        text = expand_window(tokenizer, lines, st, ed, budget)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False,
                        max_length=args.max_seq_length, truncation=True)
        tokenized.append((enc, fpath, st, ed))

    # Batch-by-length (same scheme as eval_repro.py).
    tokenized.sort(key=lambda x: x[0]["input_ids"].shape[1])
    grouped = defaultdict(list)
    for item in tokenized:
        grouped[item[0]["input_ids"].shape[1]].append(item)

    rows = []
    for length, group in tqdm(grouped.items(), desc="infer"):
        for i in range(0, len(group), args.batch_size):
            batch = group[i:i + args.batch_size]
            input_ids = torch.cat([b[0]["input_ids"] for b in batch], dim=0).to("cuda:0")
            attention_mask = torch.cat([b[0]["attention_mask"] for b in batch], dim=0).to("cuda:0")
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits[:, -1, :2].to(torch.float32), dim=-1)[:, 1]
            for (_, fpath, st, ed), prob in zip(batch, probs.cpu().tolist()):
                rows.append((fpath, st, ed, prob))

    rows.sort(key=lambda r: r[3], reverse=True)
    n_flagged = sum(1 for r in rows if r[3] > args.threshold)

    with open(args.output, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "start_line", "end_line", "prob_muf", "flagged"])
        for fpath, st, ed, prob in rows:
            w.writerow([fpath, st, ed, f"{prob:.6f}", int(prob > args.threshold)])

    print(f"\nScored {len(rows)} functions; "
          f"{n_flagged} flagged as MUF at threshold {args.threshold}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
