# Artifact Evaluation Instructions

**Paper:** "The Illusion of Rust Safety: Detecting Modular Unsafe Functions with LLMs" (CCS 2026 #1938)

**Estimated time:**
- Kick-the-tires: **~20 min** (E1, one A100-80GB)
- Full reproduction: **~2.5 h** on 4 GPUs (E2) + **~1 h** PoC eval (E3) + **~2 h** baselines (E4)

---

## Overview

This artifact reproduces the main claims of the paper:

| Claim | Result | Experiment |
|------|--------|------------|
| **C1** Fine-tuned Llama 3.2 3B classifier achieves AUPRC ≈ 0.82 on MUF detection (paper Fig. 2 / Table 2) | AUPRC ≈ 0.764 on the fixed split (≈0.82 on natural distribution) | E1 + E2 |
| **C2** Fine-tuned PoC generator produces valid PoCs demonstrating the root causes of MUFs (paper Table 4: 19/22 in-the-wild cases) | safe-caller / compile / Miri-UB metrics on held-out PoC test set | E3 |
| **C3** Coin substantially outperforms vanilla open-source LLMs (Llama 3.2 3B / 3.1 8B / Qwen3 4B; paper Fig. 2) and GPT-4o / Claude-3.7 even with few-shot context (paper Tables 2–3) | AUPRC ≈ 0.02–0.03 for vanilla open-source; precision (unsafe) ≤ 7 % for API models | E4 |

The artifact contains:
- Fine-tuned **Llama 3.2 3B classifier LoRA** at `model/llama3.2/`
- Fine-tuned **Llama 3.2 3B PoC generator LoRA** at `model/llama3.2_poc/`
- Self-contained evaluation scripts in `code/`
- Convenience launchers in `scripts/`
- Sample test data; full test sets via the AE submission system

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 24 GB (single-GPU inference) | 4 × 80 GB A100 (fast full run) |
| CPU RAM   | 32 GB | 64 GB |
| Disk      | 20 GB | 50 GB |

All reported results were measured on NVIDIA A100-80GB GPUs.

---

## Step 0: Prerequisites

### 0.1 Reassemble split model files

Model weights are split into ≤50 MB parts for repository compatibility.
**Run this once before any evaluation:**

```bash
bash model/reassemble.sh
```

Expected output:
```
Reassembling llama3.2 classifier adapter...
  -> model/llama3.2/adapter_model.safetensors (93M)
Reassembling llama3.2_poc PoC generator adapter...
  -> model/llama3.2_poc/adapter_model.safetensors (...)
Done.
```

### 0.2 Set up the Python environment

> **Critical:** The classifier evaluation requires **unsloth 2025.2.15** and **transformers 4.49.0** exactly.
> Newer unsloth versions silently truncate sequences and drop AUPRC from ~0.76 to ~0.32.
> See [Why versions matter](#why-package-versions-matter).

```bash
bash scripts/0_setup_env.sh
```

This creates `./env/` with the pinned package versions. To verify:

```bash
./env/bin/python -c "import unsloth; print(unsloth.__version__)"
# Expected: 2025.2.15
```

### 0.3 Obtain the test datasets

The full datasets are provided via the AE submission system:

| File | Use | Size |
|------|-----|------|
| `coin_test.pkl` (319,652 labeled functions) | E1, E2 | ~120 MB |
| `poc_test.jsonl` (held-out PoC examples) | E3 | <1 MB |
| `poc_train_v2.jsonl` (verified training PoCs) | E3 re-train (optional) | <5 MB |

Substitute the actual paths for `/path/to/...` in the commands below.

For smoke-testing **without the full data**, `data/coin_test.pkl.sample.part_*`
can be reassembled (`cat data/coin_test.pkl.sample.part_* > data/coin_test.pkl.sample`)
and passed to E1 with `--n 500` for a quick functional check.

---

## E1: Kick-the-Tires — Classifier Smoke Test (~20 min)

Verifies environment and model loading on an 8,000-example stratified
sample. Supports claim (C1).

```bash
bash scripts/1_smoke_test.sh /path/to/coin_test.pkl
```

Or, equivalently:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=./env/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH \
  ./env/bin/python code/eval_repro.py \
  --checkpoint model/llama3.2 \
  --data /path/to/coin_test.pkl \
  --n 8000 \
  --output /tmp/eval_smoke.json
```

**Expected output:**
```
AUPRC: 0.7529
Precision safe=0.9943 unsafe=0.4497
Recall    safe=0.9747 unsafe=0.8050
F1        safe=0.9844 unsafe=0.5771
```

If AUPRC ≈ 0.30–0.35, the unsloth version is wrong — recheck Step 0.2.

---

## E2: Full Classifier Evaluation (~2.5 h on 4 GPUs)

Reproduces AUPRC on the complete 319,652-sample test set. Supports claim (C1).

### Option A: 4-GPU sharding (~2.5 hours)

```bash
bash scripts/2_full_eval.sh /path/to/coin_test.pkl
```

This launches one shard per GPU (0–3 by default), waits for all four,
and prints aggregate AUPRC + precision-at-recall-0.8.

### Option B: Single GPU (~10 hours)

```bash
NUM_SHARDS=1 bash scripts/2_full_eval.sh /path/to/coin_test.pkl
```

### Expected results

| Metric | Expected | 
|--------|----------|
| **AUPRC** | **0.764~0.822** | 
| Recall (unsafe) @ t=0.7 | 0.838 | 
| Precision (unsafe) @ t=0.7 | 0.459 | 
| Precision (unsafe) @ recall=0.80 | 0.610 | 

---

## E3: PoC Generator Evaluation (~1 hour)

Evaluates the fine-tuned PoC generator on a held-out PoC test set with
three automated metrics. Supports claim (C2).

```bash
bash scripts/3_poc_eval.sh /path/to/poc_test.jsonl
```

Or directly:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=./env/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH \
  ./env/bin/python code/gen_poc.py --eval \
  --model_dir model/llama3.2_poc \
  --data /path/to/poc_test.jsonl \
  --n 20 --output /tmp/poc_eval.json
```

**Metrics reported:**

| Metric | Meaning |
|--------|---------|
| `safe_caller` | Generated `fn main()` contains no `unsafe` blocks (the model respected the safety constraint) |
| `compiles` | `cargo build` succeeds within 60 s |
| `ub_detected` | `cargo +nightly miri run` reports Undefined Behavior |

`compiles` and `ub_detected` require Rust + Cargo (+ nightly Miri).
Install Miri once with:
```bash
rustup toolchain install nightly
rustup component add miri --toolchain nightly
```
If Cargo is unavailable, the script still reports `safe_caller`.

### Re-train the PoC generator from scratch (optional, ~3 h on one A100)

```bash
TRAIN=1 TRAIN_DATA=/path/to/poc_train_v2.jsonl \
  bash scripts/3_poc_eval.sh /path/to/poc_test.jsonl
```

This calls `code/train_poc_generator.py`, which uses unsloth+SFTTrainer
to fine-tune a Llama 3.2 3B LoRA for ~20 epochs.

### Single-shot generation

```bash
./env/bin/python code/gen_poc.py --infer \
  --model_dir model/llama3.2_poc \
  --function_text "$(cat my_function.rs)" \
  --category "logical memory controls"
```

Produces an explanation + `Cargo.toml` + `src/main.rs` PoC.

---

## E4: Baseline Comparison (~2 h)

Reproduces the comparison against vanilla open-source LLMs (paper Fig. 2)
and closed-source GPT-4o / Claude-3.7 (paper Tables 2–3). Supports claim (C3).

### 4.1 Open-source baselines (Llama 3.2 3B / 3.1 8B / Qwen3 4B)

`code/eval_open_baseline.py` loads each vanilla model via standard
HuggingFace `transformers` (no unsloth, no LoRA), runs a single forward
pass for each test prompt, and extracts the logits of the
`"Yes"` and `"No"` tokens to compute an unsafe probability — directly
comparable to the Coin classifier's AUPRC.

```bash
bash scripts/4_baseline_eval.sh /path/to/coin_test.pkl open
```

Or one model at a time:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=./env/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH \
  ./env/bin/python code/eval_open_baseline.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --data /path/to/coin_test.pkl \
  --n 8000 \
  --output /tmp/baseline_llama32.json
```

Llama models are gated on HuggingFace — set `HF_TOKEN` or run
`huggingface-cli login` once.

**Expected results (paper Fig. 2; 8K stratified sample):**

| Model | AUPRC | Paper AUPRC |
|-------|-------|-------------|
| Llama 3.2 3B (vanilla) | ~0.02 | 0.021 |
| Llama 3.1 8B (vanilla) | ~0.02 | 0.019 |
| Qwen3 4B (vanilla) | ~0.03 | 0.030 |
| **Coin (Llama 3.2 3B, fine-tuned)** | **~0.75** (E1) | 0.822 |

### 4.1.1 One-shot script: run all open-source models and draw Fig. 2

`scripts/5_pr_curves.sh` evaluates Coin and every vanilla open-source
baseline on the same stratified test sample, then renders a single
precision–recall figure overlaying all curves (paper Figure 2).

A pre-rendered version of this figure from our own run on n=8000 is
checked in at [`figures/pr_curves.png`](figures/pr_curves.png) (PDF
sibling: [`figures/pr_curves.pdf`](figures/pr_curves.pdf)). Reviewers
can compare their re-rendered output to this reference; the AUPRC table
and run conditions for the checked-in figure are in
[`figures/summary.txt`](figures/summary.txt).

```bash
bash scripts/5_pr_curves.sh /path/to/coin_test.pkl
```

Each per-model run pickles its `(probs, labels)` to
`/tmp/coin_pr_curves/<model>_probs.pkl`; the plotter
`code/draw_pr_curves.py` then reads every sidecar and writes both
`pr_curves.png` and `pr_curves.pdf` to the same directory, plus a
`summary.txt` listing each model's AUPRC.

Environment overrides:

| Variable | Default | Purpose |
|----------|---------|---------|
| `N` | 8000 | stratified test sample size |
| `GPU` | 0 | CUDA device index |
| `OUT_DIR` | /tmp/coin_pr_curves | output directory |
| `SKIP` | (none) | comma-separated substrings; skip matching HF IDs |

Already-completed runs are cached by sidecar pickle, so re-running the
script after adding a new model only evaluates the missing ones. To
add a model, edit `run_open_baseline ... ` lines near the end of
`scripts/5_pr_curves.sh`.

To draw the figure from existing pickles without re-running any model:

```bash
./env/bin/python code/draw_pr_curves.py \
  --inputs Coin:/tmp/coin_pr_curves/coin_probs.pkl \
           'Llama 3.2 3B:/tmp/coin_pr_curves/open_llama32_3b_probs.pkl' \
           'Llama 3.1 8B:/tmp/coin_pr_curves/open_llama31_8b_probs.pkl' \
           'Qwen3 4B:/tmp/coin_pr_curves/open_qwen3_4b_probs.pkl' \
  --output /tmp/coin_pr_curves/pr_curves.png
```

The plotter auto-detects per-sample pickle shapes (Coin's
`(prob_safe, prob_unsafe)` pairs vs. the baselines' flat
`prob_unsafe` arrays) and highlights Coin's curve in red/solid.

### 4.2 Closed-source API drivers (GPT-4o / Claude-3.7)

`code/eval_api_baseline.py` is a provider-agnostic driver supporting
both OpenAI and Anthropic. Few-shot prompts (Table 2) and Best-of-K
queries (Table 3) are controlled by command-line flags.

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

bash scripts/4_baseline_eval.sh /path/to/coin_test.pkl api
```

The launcher iterates over the configurations in the paper:
N ∈ {0, 1, 3, 5} few-shot for Best@1 (Table 2), and N=1 with K ∈ {3, 5}
Best-of-K (Table 3). Each configuration runs against ~200 stratified
test examples (override with `N_API=...`) to control API cost.

For one configuration directly:

```bash
./env/bin/python code/eval_api_baseline.py \
  --provider openai --model gpt-4o \
  --data /path/to/coin_test.pkl \
  --n 200 \
  --shots_file data/shots.jsonl --shots_n 3 \
  --best_of 1 \
  --output /tmp/gpt4o_n3.json
```

**Few-shot examples:** `data/shots.jsonl` contains a tiny sample
(5 labeled examples) to make the script runnable out of the box.
The paper's experiments used examples drawn from `D_muf`; you can
generate a larger shots file by writing one JSON object per line in
the format documented at the top of `eval_api_baseline.py`.

**Expected results (paper Tables 2–3; ~200 samples; precision/recall on
the unsafe class):**

| Model | N (shots) | Best of K | Precision | Recall | Paper |
|-------|-----------|-----------|-----------|--------|-------|
| GPT-4o | 1 | 1 | ~5 % | ~21 % | 4.6 % / 21.4 % |
| GPT-4o | 3 | 1 | ~4 % | ~21 % | 3.7 % / 21.4 % |
| GPT-4o | 5 | 1 | ~4 % | ~21 % | 3.9 % / 21.4 % |
| Claude-3.7 | 1 | 1 | ~4 % | ~7 % | 4.2 % / 7.1 % |
| Claude-3.7 | 3 | 1 | ~7 % | ~12 % | 6.5 % / 11.9 % |
| Claude-3.7 | 5 | 1 | ~7 % | ~12 % | 6.9 % / 11.9 % |
| **Coin** | — | — | **63.7 %** | **80.4 %** | **63.71 % / 80.42 %** |

Numbers will fluctuate slightly with the stratified-200 sample and API
non-determinism but should remain in the same low range — the point of
the comparison is the order-of-magnitude precision gap vs. Coin.

> **API cost note:** A full GPT-4o + Claude-3.7 sweep of all eight
> configurations × ~200 examples is roughly 3,200 API calls; budget
> a few US dollars per provider. Reduce `N_API` if needed.

---

## Optional: Run Classifier on a New Crate (Reusability)

```bash
cd code
../env/bin/python infer_batch.py \
  --crate_dir /path/to/rust/crate \
  --model_dir ../model/llama3.2
```

Output: a CSV listing each candidate function with its predicted MUF probability.

---

## Why Package Versions Matter

`eval_repro.py` loads the model with:
```python
model, tokenizer = FastLanguageModel.from_pretrained(checkpoint_path)
```
No `max_seq_length` argument is passed. **Unsloth 2025.2.15** reads
`max_seq_length=8192` from the checkpoint config automatically.
**Unsloth ≥ 2026.x** changed this behavior to default to a shorter
length when the argument is omitted, silently truncating long Rust code
snippets and degrading AUPRC from ~0.76 to ~0.32.

| Unsloth version | max_seq_length used | AUPRC (8K sample) |
|-----------------|---------------------|-------------------|
| 2026.x (new default) | 512–2048 | ~0.32 |
| **2025.2.15 (required)** | **8192 (from config)** | **~0.75** |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| AUPRC ≈ 0.30–0.35 | Wrong unsloth version | Re-run `scripts/0_setup_env.sh` |
| `ImportError: bitsandbytes` | Missing CUDA lib path | Set `LD_LIBRARY_PATH` as in scripts |
| OOM during eval | Long 8192-tok sequences | Reduce `--n` for smoke test, or shard |
| `cargo: command not found` (E3) | Rust not installed | E3 still reports `safe_caller`; install rustup for `compiles`/`ub` |
| Slow inference (>10 s/batch) | SDPA fallback to eager | Expected on some GPUs; use sharding |
