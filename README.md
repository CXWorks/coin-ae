# Coin — Artifact

**The Illusion of Rust Safety: Detecting Modular Unsafe Functions with LLMs**

This artifact reproduces the two main results of the paper:

| Claim | Artifact | Reproduction step |
|-------|----------|--------|
| **C1** — Llama 3.2 3B classifier achieves AUPRC ≈ 0.82 on MUF detection | `model/llama3.2/` | E1 (smoke test) + E2 (full eval) |
| **C2** — PoC generator produces valid PoCs for 19/22 in-the-wild MUF bugs | `model/llama3.2_poc/` | E3 (PoC eval) |

See [ARTIFACT_EVALUATION.md](ARTIFACT_EVALUATION.md) for full step-by-step instructions.

## Layout

```
coin-ae-final/
├── ARTIFACT_EVALUATION.md       # Detailed AE instructions (this is the main doc)
├── coin-ae-appendix.tex         # LaTeX appendix for the AE submission
├── code/
│   ├── eval_repro.py            # E1/E2: classifier evaluation (Llama 3.2 3B)
│   ├── infer_batch.py           # Reusability: run classifier on a new crate
│   ├── train_poc_generator.py   # Optional: re-train the PoC generator LoRA
│   ├── gen_poc.py               # E3: PoC generator inference + evaluation
│   ├── threshold.py             # PAC-based threshold calibration
│   ├── llm_final.py             # Baseline: GPT-4o / Claude-3.7 few-shot
│   └── unsafe_collect.py        # Utility: collect MUF candidates from a crate
├── model/
│   ├── llama3.2/                # Fine-tuned classifier LoRA (Llama 3.2 3B base)
│   ├── llama3.2_poc/            # Fine-tuned PoC generator LoRA
│   └── reassemble.sh            # Concatenate the .part_* files; run once
├── data/
│   └── coin_test.pkl.sample.*   # Reassemblable sample test split
├── prompts/                     # Prompt templates used during training/inference
├── custom_rustc_patch/          # rustc 1.83.0-dev patch for safe-candidate extraction
└── scripts/
    ├── 0_setup_env.sh           # Conda env with the pinned package versions
    ├── 1_smoke_test.sh          # E1: 8K-sample classifier smoke test (~20 min)
    ├── 2_full_eval.sh           # E2: full 319K classifier eval (~2.5 h on 4 GPUs)
    └── 3_poc_eval.sh            # E3: PoC generator evaluation
```

## Quick start

```bash
# 1. Reassemble split model files (one time)
bash model/reassemble.sh

# 2. Create the conda env (requires unsloth==2025.2.15)
bash scripts/0_setup_env.sh

# 3. Smoke test (8K samples; expected AUPRC ≈ 0.75)
bash scripts/1_smoke_test.sh /path/to/coin_test.pkl
```

## Critical: unsloth version

The classifier evaluation requires **unsloth 2025.2.15** exactly. Newer
versions silently truncate sequences and degrade AUPRC from ~0.76 to ~0.32.
The setup script pins this; do not upgrade. 

## Datasets

The sample data under `data/` is for quick smoke-testing only. The full
test sets (`coin_test.pkl`, `poc_train_v2.jsonl`, `poc_test.jsonl`) are
provided separately via the AE submission system due to size.

## Licenses

- Code: MIT
- `model/llama3.2/` and `model/llama3.2_poc/`: Llama 3.2 Community License
  (LoRA adapters on top of `unsloth/Llama-3.2-3B-bnb-4bit`)
- Anonymous artifact repository:
  <https://anonymous.4open.science/r/coin-ae-E308>
