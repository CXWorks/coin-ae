"""
gen_poc.py - Inference and evaluation for the Coin PoC generator (Llama 3.2 3B LoRA).

Two modes:

  --infer        Generate a PoC for a single MUF function (interactive).
  --eval         Run automated evaluation on data/poc_test.jsonl.

Usage:
    # Generate one PoC
    python gen_poc.py --infer \
        --model_dir ../model/llama3.2_poc \
        --function_text "$(cat my_function.txt)" \
        --category "logical memory controls"

    # Batch evaluation: safe_caller / compiles / ub_detected
    python gen_poc.py --eval \
        --model_dir ../model/llama3.2_poc \
        --data /path/to/poc_test.jsonl \
        --n 20 --output /tmp/poc_eval.json

Evaluation metrics (per generated PoC):
  safe_caller  - fn main() in src/main.rs contains NO unsafe blocks
  compiles     - cargo build succeeds within 60s
  ub_detected  - cargo +nightly miri run reports Undefined Behavior

Miri (optional) requires:
    rustup toolchain install nightly
    rustup component add miri --toolchain nightly
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["WANDB_DISABLED"] = "true"

import torch
from unsloth import FastLanguageModel


RESPONSE_TEMPLATE = "### Response:\n"

CATEGORIES = [
    "logical requirement",
    "ffi",
    "logical memory controls",
    "sharing status",
    "steal reference",
    "embedding memory mapping",
    "hardware feature",
]

UB_MARKERS = [
    "undefined behavior", "use-after-free", "out-of-bounds",
    "invalid value", "stacked borrows", "borrow stack",
    "dangling reference",
]


def format_prompt(function_text: str, category: str) -> str:
    return (
        f"### Instruction:\n"
        f"The following Rust function is a Modular Unsafe Function (MUF) of category "
        f"\"{category}\".\n"
        f"Write a minimal PoC (Cargo.toml + main.rs) that demonstrates undefined behavior "
        f"by violating its invariant.\n\n"
        f"Function:\n"
        f"```rust\n{function_text}\n```\n"
        f"{RESPONSE_TEMPLATE}"
    )


def parse_output(text: str) -> dict:
    def extract(label: str) -> str:
        pat = rf'### {re.escape(label)}:\s*\n+```(?:\w*)\n(.*?)```'
        m = re.search(pat, text, re.S)
        return m.group(1).strip() if m else ""

    expl = ""
    if "### Explanation:" in text:
        expl = text.split("### Explanation:", 1)[1].split("###", 1)[0].strip()

    return {
        "explanation":    expl,
        "poc_cargo_toml": extract("Cargo.toml"),
        "poc_main_rs":    extract("src/main.rs"),
    }


def has_unsafe_in_main(main_rs: str) -> bool:
    idx = main_rs.find("fn main(")
    if idx == -1:
        return "unsafe" in main_rs
    return bool(re.search(r'\bunsafe\b', main_rs[idx:]))


def run_compile_and_miri(cargo_toml: str, main_rs: str, timeout: int) -> tuple:
    if not cargo_toml or not main_rs:
        return False, False, "MISSING_CODE"
    with tempfile.TemporaryDirectory(prefix="poc_eval_") as tmp:
        p = Path(tmp)
        (p / "Cargo.toml").write_text(cargo_toml)
        (p / "src").mkdir()
        (p / "src/main.rs").write_text(main_rs)
        try:
            build = subprocess.run(
                ["cargo", "build"], cwd=tmp,
                capture_output=True, text=True, timeout=60,
            )
            if build.returncode != 0:
                return False, False, build.stderr[:300]
            miri = subprocess.run(
                ["cargo", "+nightly", "miri", "run"], cwd=tmp,
                capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "MIRIFLAGS": "-Zmiri-disable-isolation"},
            )
            log = miri.stdout + miri.stderr
            ub = any(m in log.lower() for m in UB_MARKERS)
            return True, ub, log[:300]
        except subprocess.TimeoutExpired:
            return False, False, "TIMEOUT"
        except FileNotFoundError:
            return False, False, "CARGO_NOT_INSTALLED"
        except Exception as e:
            return False, False, str(e)[:300]


def load_model(model_dir: str):
    print(f"Loading PoC generator from {model_dir}...", file=sys.stderr)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate(model, tokenizer, function_text: str, category: str, max_new_tokens: int = 1024) -> str:
    prompt = format_prompt(function_text, category)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return full[len(prompt):] if full.startswith(prompt) else full


def cmd_infer(args):
    if args.category not in CATEGORIES:
        print(f"Warning: category '{args.category}' not in known set: {CATEGORIES}",
              file=sys.stderr)
    model, tokenizer = load_model(args.model_dir)
    out = generate(model, tokenizer, args.function_text, args.category)
    print(out)


def cmd_eval(args):
    records = []
    with open(args.data) as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("function_text") or not rec.get("category"):
                continue
            records.append(rec)
    if args.n > 0:
        records = records[:args.n]
    print(f"Evaluating on {len(records)} examples", file=sys.stderr)

    model, tokenizer = load_model(args.model_dir)

    results = []
    stats = {"safe_caller": 0, "compiles": 0, "ub_detected": 0, "total": len(records)}

    for i, rec in enumerate(records):
        raw = generate(model, tokenizer, rec["function_text"], rec["category"])
        parsed = parse_output(raw)
        main_rs = parsed["poc_main_rs"]
        safe = (not has_unsafe_in_main(main_rs)) if main_rs else False
        compiled, ub, log = run_compile_and_miri(
            parsed["poc_cargo_toml"], main_rs, timeout=args.timeout,
        )
        stats["safe_caller"]  += int(safe)
        stats["compiles"]     += int(compiled)
        stats["ub_detected"]  += int(ub)
        results.append({
            "idx": i, "category": rec["category"],
            "safe_caller": safe, "compiles": compiled, "ub_detected": ub,
            "log": log,
        })
        print(f"[{i+1}/{len(records)}] cat={rec['category']:30s} "
              f"safe={safe} compiles={compiled} ub={ub}", file=sys.stderr)

    summary = {
        "model_dir": args.model_dir, "data": args.data,
        "n": len(records),
        "safe_caller_rate":  stats["safe_caller"]  / max(1, len(records)),
        "compile_rate":      stats["compiles"]     / max(1, len(records)),
        "ub_detected_rate":  stats["ub_detected"]  / max(1, len(records)),
        "counts": stats,
    }
    print("\n=== Summary ===")
    print(f"safe_caller : {stats['safe_caller']}/{len(records)} "
          f"({summary['safe_caller_rate']:.2%})")
    print(f"compiles    : {stats['compiles']}/{len(records)} "
          f"({summary['compile_rate']:.2%})")
    print(f"ub_detected : {stats['ub_detected']}/{len(records)} "
          f"({summary['ub_detected_rate']:.2%})")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"summary": summary, "per_example": results}, f, indent=2)
        print(f"\nSaved to {args.output}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=False)

    p.add_argument("--infer", action="store_true",
                   help="Generate a single PoC for the given function.")
    p.add_argument("--eval", action="store_true",
                   help="Run automated evaluation on a JSONL file.")
    p.add_argument("--model_dir", default="../model/llama3.2_poc")

    p.add_argument("--function_text", help="(--infer) MUF source code")
    p.add_argument("--category", help="(--infer) MUF category label")

    p.add_argument("--data", help="(--eval) path to poc_test.jsonl")
    p.add_argument("--n", type=int, default=0,
                   help="(--eval) limit to N examples (0 = all)")
    p.add_argument("--output", help="(--eval) JSON output path")
    p.add_argument("--timeout", type=int, default=90,
                   help="(--eval) Miri timeout in seconds")

    args = p.parse_args()
    if args.infer and args.eval:
        sys.exit("Choose exactly one of --infer or --eval")
    if args.infer:
        if not args.function_text or not args.category:
            sys.exit("--infer requires --function_text and --category")
        cmd_infer(args)
    elif args.eval:
        if not args.data:
            sys.exit("--eval requires --data")
        cmd_eval(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
