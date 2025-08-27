import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ─── Config ───────────────────────────────────────────────────────────────────
model_name       = "outputs_sec/checkpoint-11000/"
max_seq_length   = 8192
device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD_UNSAFE = 0.85

# This must match the column in your raw dataset.
# If your dataset doesn't have a real “file path” column, we’ll create one from the index.
FILEPATH_COLUMN = "file_location"

# Use a batch size that’s a multiple of 8 so each GPU gets an equal share
BATCH_SIZE = 1


def compute_metrics_from_logits(logits_array: np.ndarray, labels_array: np.ndarray):
    """
    Given all logits (N×2) and true labels (N,), compute predictions via
    softmax + threshold, then return accuracy/precision/recall/f1 + confusion‐matrix counts.
    """
    probs = F.softmax(torch.from_numpy(logits_array), dim=1).numpy()
    preds = (probs[:, 1] >= THRESHOLD_UNSAFE).astype(int)

    accuracy  = accuracy_score(labels_array, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_array,
        preds,
        average=None,
        labels=[0, 1],
        zero_division=0
    )
    conf = confusion_matrix(labels_array, preds, labels=[0, 1])

    return {
        "accuracy":         accuracy,
        "precision_safe":   precision[0],
        "precision_unsafe": precision[1],
        "recall_safe":      recall[0],
        "recall_unsafe":    recall[1],
        "f1_safe":          f1[0],
        "f1_unsafe":        f1[1],
        "true_negatives":   int(conf[0][0]),
        "false_positives":  int(conf[0][1]),
        "false_negatives":  int(conf[1][0]),
        "true_positives":   int(conf[1][1]),
        "macro_precision":  float(np.mean(precision)),
        "macro_recall":     float(np.mean(recall)),
        "macro_f1":         float(np.mean(f1)),
    }


def formatting_prompts_func(example):
    prompt = """Here is a Rust code and please check if the function starting with `>` is safe or unsafe:
    {}

    Is this function unsafe?
    """

    if isinstance(example["function_text"], str):
        return prompt.format(example["function_text"])

    texts = []
    for i in range(len(example["function_text"])):
        texts.append(prompt.format(example["function_text"][i]))
    return texts


def process_dataset_for_classification(dataset, tokenizer, max_length):
    """
    Tokenize each example’s `function_text` → input_ids/attention_mask,
    keep `label`, and keep FILEPATH_COLUMN (e.g. `file_location`).
    """
    def tokenize_function(examples):
        texts = formatting_prompts_func(examples)
        tokenized = tokenizer(
            texts if isinstance(texts, list) else [texts],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        tokenized["labels"] = examples["label"]
        tokenized[FILEPATH_COLUMN] = examples[FILEPATH_COLUMN]
        return tokenized

    # Drop everything except “label” and FILEPATH_COLUMN from the raw columns
    to_remove = [col for col in dataset.column_names if col not in {FILEPATH_COLUMN, "label"}]
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=to_remove,
        num_proc=64
    )
    return tokenized_dataset


def collate_fn(batch):
    """
    Given a list of examples (each example has 'input_ids', 'attention_mask', 'labels', and FILEPATH_COLUMN),
    stack the tensors and gather file_path into a Python list.
    """
    input_ids      = torch.stack([x["input_ids"] for x in batch], dim=0)            # (batch_size, seq_len)
    attention_mask = torch.stack([x["attention_mask"] for x in batch], dim=0)
    labels         = torch.tensor([int(x["labels"].item()) for x in batch], dtype=torch.long)
    file_paths     = [x[FILEPATH_COLUMN] for x in batch]

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        FILEPATH_COLUMN:  file_paths
    }


if __name__ == "__main__":
    # ─── 1) Load tokenizer & Llama 3.2 3B model ────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    # Wrap in DataParallel over all available GPUs (0–7)
    if torch.cuda.device_count() > 1:
        cuda_ids = list(range(min(torch.cuda.device_count(), 8)))
        model = nn.DataParallel(model, device_ids=cuda_ids)
    model.eval()

    # ─── 2) Load raw dataset ───────────────────────────────────────────────────────
    raw_datasets = load_dataset("data/coinstd", "caller")
    test_raw = raw_datasets["test"]

    # ─── 3) Ensure “file_location” exists. If not, inject index as file_location ───
    if FILEPATH_COLUMN not in test_raw.column_names:
        test_raw = test_raw.map(lambda example, idx: {FILEPATH_COLUMN: idx}, with_indices=True)

    # ─── 4) Tokenize, carrying over “file_location” and “label” ───────────────────
    test_tokenized = process_dataset_for_classification(test_raw, tokenizer, max_seq_length)

    # Tell HuggingFace to return Tensors for input_ids, attention_mask, labels.
    # FILEPATH_COLUMN stays as a Python object (int or str).
    test_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", FILEPATH_COLUMN])

    # ─── 5) Build DataLoader for 8‐GPU inference ───────────────────────────────────
    dataloader = DataLoader(
        test_tokenized,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True
    )

    all_logits       = []
    all_labels       = []
    all_preds        = []
    all_filepaths    = []
    all_prob_safe    = []
    all_prob_unsafe  = []

    # ─── 6) Run inference in batched loop with tqdm ─────────────────────────────────
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference on 8 GPUs"):
            input_ids       = batch["input_ids"].to(device)         # (B, seq_len)
            attention_mask  = batch["attention_mask"].to(device)    # (B, seq_len)
            labels         = batch["labels"].cpu().numpy()          # (B,)
            file_paths     = batch[FILEPATH_COLUMN]                 # list of length B

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits.cpu().numpy()                  # shape (B, 2)
            probs   = F.softmax(torch.from_numpy(logits), dim=1).numpy()

            preds = (probs[:, 1] >= THRESHOLD_UNSAFE).astype(int)

            # Collect per‐item in this batch
            for i in range(logits.shape[0]):
                all_logits.append(logits[i])
                all_labels.append(int(labels[i]))
                all_preds.append(int(preds[i]))
                all_filepaths.append(file_paths[i])
                all_prob_safe.append(float(probs[i, 0]))
                all_prob_unsafe.append(float(probs[i, 1]))

    # ─── 7) Stack logits & compute metrics ─────────────────────────────────────────
    logits_array = np.stack(all_logits, axis=0)       # shape (N, 2)
    labels_array = np.array(all_labels, dtype=int)    # shape (N,)

    # ─── 8) Save to CSV: file_location, true_label, pred_label, prob_safe, prob_unsafe ─
    df = pd.DataFrame({
        "file_location": all_filepaths,
        "true_label":    all_labels,
        "pred_label":    all_preds,
        "prob_safe":     all_prob_safe,
        "prob_unsafe":   all_prob_unsafe
    })
    os.makedirs("outputs_8b_classifier_infer", exist_ok=True)
    out_csv = "outputs_8b_classifier_infer/results_with_file_paths.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n➡️  Saved predictions + file paths to: {out_csv}\n")

    # ─── 9) Print metrics ──────────────────────────────────────────────────────────
    metrics = compute_metrics_from_logits(logits_array, labels_array)
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")

