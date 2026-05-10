"""
Fine-tune Llama 3.2 3B for PoC generation using Unsloth + SFTTrainer (trl>=0.19).

Reads data/poc_dataset/poc_train.jsonl and trains a LoRA adapter to generate
PoC Rust code given a MUF function + category.

Usage:
    conda run -n rustffi python3 ft_poc.py

Checkpoint saved to: outputs_poc/
"""
import json
import os
import random

os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import unsloth  # must be first to patch trl/transformers
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

TRAIN_FILE = "data/poc_dataset/poc_train.jsonl"
OUTPUT_DIR = "outputs_poc"
MAX_SEQ_LENGTH = 4096
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


def format_example(rec: dict) -> str:
    instruction = (
        f"### Instruction:\n"
        f"The following Rust function is a Modular Unsafe Function (MUF) of category "
        f'"{rec["category"]}".\n'
        f"Write a minimal PoC (Cargo.toml + main.rs) that demonstrates undefined behavior "
        f"by violating its invariant.\n\n"
        f"Function:\n"
        f"```rust\n{rec['function_text']}\n```\n"
    )
    response = (
        f"{RESPONSE_TEMPLATE}"
        f"### Explanation:\n{rec['explanation']}\n\n"
        f"### Cargo.toml:\n"
        f"```toml\n{rec['poc_cargo_toml']}\n```\n\n"
        f"### src/main.rs:\n"
        f"```rust\n{rec['poc_main_rs']}\n```\n\n"
        f"### Verification command:\n"
        f"```\ncargo +nightly miri run\n```"
    )
    return instruction + response


def load_dataset_from_jsonl(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            if not rec.get("verified"):
                continue
            if not rec.get("poc_main_rs") or not rec.get("poc_cargo_toml"):
                continue
            records.append({"text": format_example(rec), "category": rec["category"]})

    print(f"Loaded {len(records)} verified PoC training examples")
    from collections import Counter
    for cat, cnt in sorted(Counter(r["category"] for r in records).items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")

    random.shuffle(records)
    return Dataset.from_list(records)


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-bnb-4bit",
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=True,
    )

    dataset = load_dataset_from_jsonl(TRAIN_FILE)
    n = len(dataset)
    if n < 10:
        raise ValueError(f"Too few training examples ({n}). Need at least 10 verified PoCs.")

    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]
    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # trl>=0.19: SFTConfig replaces TrainingArguments; processing_class replaces tokenizer.
    # dataset_text_field and max_seq_length live on SFTConfig.
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            num_train_epochs=20,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            seed=42,
        ),
    )

    print("Starting PoC generator fine-tuning...")
    trainer.train()

    final_path = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
