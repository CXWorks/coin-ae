---
base_model: unsloth/Llama-3.2-3B-bnb-4bit
library_name: peft
task: PoC generation for Modular Unsafe Functions in Rust
---

# Coin PoC Generator — Llama 3.2 3B LoRA

Fine-tuned LoRA adapter for generating Proof-of-Concept (PoC) exploit code for
Modular Unsafe Functions (MUFs) detected by the Coin classifier.

## Model Details

- **Base model**: `unsloth/Llama-3.2-3B-bnb-4bit` (4-bit quantized via Unsloth)
- **Adapter type**: LoRA (r=16, alpha=16)
- **Task**: Given a flagged safe Rust function and its MUF category, generate a
  minimal Cargo.toml + main.rs that triggers undefined behavior
- **Training**: ~140 epochs on verified PoC examples across 7 MUF categories
- **Hardware**: NVIDIA RTX A6000 (48 GB), ~40 GPU-hours

## Files

| File | Description |
|------|-------------|
| `adapter_config.json` | LoRA configuration |
| `adapter_model.safetensors.part_aa/ab` | Split adapter weights (reassemble before use) |
| `tokenizer.json` | Tokenizer vocabulary |
| `tokenizer_config.json` | Tokenizer settings |

Run `bash model/reassemble.sh` once to reconstruct `adapter_model.safetensors`.

## Loading

```python
from unsloth import FastLanguageModel
from peft import PeftModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-3B-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, "model/llama3.2_poc")
FastLanguageModel.for_inference(model)
```

## Prompt Format

See `prompts/poc_generation_prompt.md` for the full prompt template.

## Citation

The Illusion of Rust Safety: Detecting Modular Unsafe Functions with LLMs.
CCS 2026. https://doi.org/XXXXXXX.XXXXXXX
