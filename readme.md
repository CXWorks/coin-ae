# AE for Coin: Detecting Modular Unsafe Functions with LLMs

Artifact for CCS 2026 paper #1938 — "The Illusion of Rust Safety: Detecting Modular Unsafe Functions with LLMs".

Artifact repository: https://anonymous.4open.science/r/coin-ae-E308

## Contents

### `code/` — Training and evaluation scripts

| Script | Purpose |
|--------|---------|
| `ft1.py` | Fine-tune Llama 3.2 3B classifier (MUF detection) |
| `train_poc_generator.py` | Fine-tune Llama 3.2 3B PoC generator |
| `infer.py` / `infer_batch.py` | Run classifier inference on Rust crates |
| `eval.py` / `eval_ds.py` | Evaluate classifier precision/recall |
| `llm_final.py` | Vanilla LLM few-shot baseline (GPT-4o, Claude-3.7) |
| `threshold.py` | PAC-based thresholding for calibrated precision/recall |
| `build_eval.py` / `build_std.py` | Build evaluation datasets from Rust standard library |
| `unsafe_collect.py` | Collect safe function candidates from crate source |

Set `OPENAI_API_KEY` environment variable before running `llm_final.py`.

### `custom_rustc_patch/`

Patch diff for the customized Rust toolchain (1.83.0-dev) used to extract safe
function candidates with surrounding context. Apply with:
```bash
cd /path/to/rust && git apply /path/to/rustc.patch
```

### `data/` — Sample dataset (training data withheld pending publication)

Sample splits for smoke-testing the training pipeline:
- `coin_train.pkl.sample.part_*` — training set sample
- `coin_valid.pkl.sample.part_*` — validation set sample
- `coin_test.pkl.sample.part_*` — test set sample

Reassemble each: `cat coin_train.pkl.sample.part_* > coin_train.pkl.sample`

Full training dataset (74K labeled MUF candidates) will be released upon paper publication.

### `model/` — Fine-tuned model weights

Two LoRA adapters, each split into ≤50 MB parts for GitHub compatibility.
**Run `bash model/reassemble.sh` before loading any model.**

| Directory | Model | Task |
|-----------|-------|------|
| `model/llama3.2/` | Llama 3.2 3B LoRA | MUF classifier |
| `model/llama3.2_poc/` | Llama 3.2 3B LoRA | PoC generator |
| `model/qwen/` | QWen 2.5 1.5B LoRA | MUF classifier (smaller baseline) |

Both adapters load on top of `unsloth/Llama-3.2-3B-bnb-4bit` (base model from HuggingFace).
See each model's `README.md` for loading instructions.

### `prompts/` — Prompt templates

| File | Description |
|------|-------------|
| `classifier_prompt.md` | Prompt used to fine-tune and run the MUF classifier |
| `poc_generation_prompt.md` | Prompt used to fine-tune and run the PoC generator |
| `vanilla_llm_prompt.md` | Few-shot baseline prompt for GPT-4o / Claude-3.7 |

## Quick Start

### 1. Reassemble models
```bash
bash model/reassemble.sh
```

### 2. Run the classifier on a crate
```bash
cd code
python infer_batch.py --crate_dir /path/to/crate --model_dir ../model/llama3.2
```

### 3. Generate a PoC for a flagged function
```bash
python train_poc_generator.py --infer --model_dir ../model/llama3.2_poc \
    --function_text "$(cat my_function.txt)" --category "logical memory controls"
```

### 4. Run the vanilla LLM baseline
```bash
export OPENAI_API_KEY=your_key_here
python llm_final.py
```

## Requirements

- Python 3.10+
- `unsloth`, `peft`, `trl`, `transformers`, `torch`
- Rust toolchain 1.83.0-dev (for crate analysis); see `custom_rustc_patch/`
- NVIDIA GPU with ≥24 GB VRAM for inference; ≥48 GB for fine-tuning
