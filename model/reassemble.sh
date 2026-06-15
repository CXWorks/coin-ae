#!/bin/bash
# Reassemble split model files before use.
# Run once from the repo root: bash model/reassemble.sh

set -e

echo "Reassembling llama3.2 classifier adapter..."
cat model/llama3.2/adapter_model.safetensors.part_* > model/llama3.2/adapter_model.safetensors
cat model/llama3.2/tokenizer.json.part_*            > model/llama3.2/tokenizer.json
echo "  -> model/llama3.2/adapter_model.safetensors ($(du -sh model/llama3.2/adapter_model.safetensors | cut -f1))"

echo "Reassembling llama3.2_poc PoC generator adapter..."
cat model/llama3.2_poc/adapter_model.safetensors.part_* > model/llama3.2_poc/adapter_model.safetensors
echo "  -> model/llama3.2_poc/adapter_model.safetensors ($(du -sh model/llama3.2_poc/adapter_model.safetensors | cut -f1))"

echo "Done. Load each adapter with PEFT on top of the base model"
echo "(unsloth/Llama-3.2-3B-bnb-4bit, listed in each adapter_config.json)."
