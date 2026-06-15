# Artifact Evaluation Instructions

**Paper:** "The Illusion of Rust Safety: Detecting Modular Unsafe Functions with LLMs" (CCS 2026 #1938)

**Estimated time:**
- Kick-the-tires: **~20 min** (E1, one A100-80GB)
- Full reproduction: **~2.5 h** on 4 GPUs (E2) + **~1 h** PoC eval (E3)

---

## Overview

This artifact reproduces the two main claims of the paper:

| Claim | Result | Experiment |
|------|--------|------------|
| **C1** Fine-tuned Llama 3.2 3B classifier achieves AUPRC ≈ 0.82 on MUF detection (paper Fig. 2 / Table 2) | AUPRC ≈ 0.764 on the fixed split (≈0.82 on natural distribution) | E1 + E2 |
| **C2** Fine-tuned PoC generator produces valid PoCs demonstrating the root causes of MUFs (paper Table 4: 19/22 in-the-wild cases) | safe-caller / compile / Miri-UB metrics on held-out PoC test set | E3 |

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
