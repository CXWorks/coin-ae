import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import argparse


# Function to format prompts
def formatting_prompts_func(example):
    prompt = """Here is a Rust code and please check if the function starting with `>` is safe or unsafe:
    {}

    Is this function unsafe?
    """

    if isinstance(example['function_text'], str):
        return prompt.format(example['function_text'])

    texts = []
    for i in range(len(example['function_text'])):
        texts.append(prompt.format(example['function_text'][i]))
    return texts


# Function to process dataset for classification
def process_dataset_for_classification(dataset, tokenizer, max_length):
    def tokenize_function(examples):
        # Format prompts
        texts = formatting_prompts_func(examples)

        # Tokenize
        tokenized = tokenizer(
            texts if isinstance(texts, list) else [texts],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # Add labels
        if not isinstance(texts, list):
            tokenized["labels"] = 0
        else:
            tokenized["labels"] = [0] * len(texts)

        return tokenized

    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=32  # Reduced from 64 for inference
    )

    return tokenized_dataset


def compute_metrics_with_threshold(eval_pred, threshold=0.7):
    logits, labels = eval_pred

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()

    # Apply custom threshold for predictions (0.7 for unsafe class)
    # If probability for unsafe (class 1) is >= threshold, predict unsafe, otherwise safe
    predictions = np.where(probabilities[:, 1] >= threshold, 1, 0)
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=None, labels=[0, 1],
                                                               zero_division=0)
    conf_matrix = confusion_matrix(labels, predictions, labels=[0, 1])

    results = {
        'accuracy': accuracy,
        'precision_safe': precision[0],
        'precision_unsafe': precision[1],
        'recall_safe': recall[0],
        'recall_unsafe': recall[1],
        'f1_safe': f1[0],
        'f1_unsafe': f1[1],
        'true_negatives': conf_matrix[0][0],
        'false_positives': conf_matrix[0][1],
        'false_negatives': conf_matrix[1][0],
        'true_positives': conf_matrix[1][1],
        'macro_precision': np.mean(precision),
        'macro_recall': np.mean(recall),
        'macro_f1': np.mean(f1),
        'threshold_used': threshold
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Run inference with Rust code safety classifier')
    parser.add_argument('--model_path', type=str, default="final_model_classifier.pt",
                        help='Path to the saved model')
    parser.add_argument('--threshold', type=float, default=0.85,
                        help='Classification threshold for unsafe class (default: 0.85)')
    parser.add_argument('--dataset_split', type=str, default="test",
                        choices=["train", "validation", "test"],
                        help='Dataset split to evaluate (default: test)')
    parser.add_argument('--output_file', type=str, default='test_eval.pkl', help='Path to save detailed results')
    parser.add_argument('--max_seq_length', type=int, default=8192, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer pad token: {tokenizer.pad_token}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
    )
    model.config.pad_token_id = tokenizer.pad_token_id  # Set pad token in model config
    model.to(device)
    print("Model loaded successfully")

    # Load dataset
    print(f"Loading dataset: data/coin, split: {args.dataset_split}")
    raw_datasets = load_dataset("data/coineval", "caller")
    print(len(raw_datasets["test"]))
    if args.dataset_split == "test":
        eval_dataset = raw_datasets["test"].select(range(20000, 40000))
    elif args.dataset_split == "validation":
        eval_dataset = raw_datasets["validation"]
    else:
        eval_dataset = raw_datasets["train"]
    #eval_dataset = raw_datasets["validation"]
    print(f"Dataset loaded with {len(eval_dataset)} examples")

    # Process dataset
    print("Processing dataset...")
    processed_dataset = process_dataset_for_classification(eval_dataset, tokenizer, args.max_seq_length)

    # Create training args for inference (with fixed batch size of 1 to avoid padding issues)
    inference_args = TrainingArguments(
        output_dir="outputs_inference",
        per_device_eval_batch_size=args.batch_size,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    # Create trainer for inference
    trainer = Trainer(
        model=model,
        args=inference_args,
        compute_metrics=lambda pred: compute_metrics_with_threshold(pred, threshold=args.threshold),
    )

    # Run prediction
    print(f"Running inference with threshold: {args.threshold}...")
    results = trainer.predict(test_dataset=processed_dataset)

    # Extract metrics and predictions
    metrics = results.metrics
    predictions = results.predictions
    labels = processed_dataset['labels']

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(torch.tensor(predictions), dim=1).numpy()

    # Apply threshold for final predictions
    thresholded_predictions = np.where(probabilities[:, 1] >= args.threshold, 1, 0)

    # Create detailed results
    detailed_results = []
    for i in range(len(thresholded_predictions)):
        if thresholded_predictions[i] == 1:
            detailed_results.append({
                "prediction": "unsafe" if thresholded_predictions[i] == 1 else "safe",
                "probabilities": {
                    "safe": float(probabilities[i][0]),
                    "unsafe": float(probabilities[i][1])
                },
                'function_text': eval_dataset['function_text'][i],
                'file_location': eval_dataset['file_location'][i]
            })
    print(len(detailed_results))
    import pickle
    with open(args.output_file, "wb") as fp:
        pickle.dump(detailed_results, fp)
    print('done', len(detailed_results))
    # Print metrics
    # print("\nEvaluation Results with Custom Threshold:")
    # print(f"Threshold used: {args.threshold}")
    # for metric_name, metric_value in metrics.items():
    #     if isinstance(metric_value, float):
    #         print(f"{metric_name}: {metric_value:.4f}")
    #     else:
    #         print(f"{metric_name}: {metric_value}")

    # Calculate error analysis statistics
    # safe_total = metrics['test_true_negatives'] + metrics['test_false_positives']
    # unsafe_total = metrics['test_true_positives'] + metrics['test_false_negatives']
    # print("\nError Analysis:")
    # print(
    #     f"Safe functions: {safe_total} (Correctly classified: {metrics['true_negatives']} [{metrics['true_negatives'] / safe_total * 100:.2f}%])")
    # print(
    #     f"Unsafe functions: {unsafe_total} (Correctly classified: {metrics['true_positives']} [{metrics['true_positives'] / unsafe_total * 100:.2f}%])")

    # Save detailed results if specified
    # if args.output_file:
    #     import json
    #     with open(args.output_file, 'w') as f:
    #         json.dump({
    #             "metrics": {k: float(v) if isinstance(v, np.float32) else v for k, v in metrics.items()},
    #             "detailed_results": detailed_results[:1000]  # Limit to 1000 entries to avoid large files
    #         }, f, indent=2)
    #     print(f"Detailed results saved to {args.output_file}")
    #
    #     # If there are more than 1000 examples, also save stats about incorrectly classified examples
    #     if len(detailed_results) > 1000:
    #         incorrect_examples = [r for r in detailed_results if not r["correct"]]
    #         false_positives = [r for r in detailed_results if
    #                            r["ground_truth"] == "safe" and r["prediction"] == "unsafe"]
    #         false_negatives = [r for r in detailed_results if
    #                            r["ground_truth"] == "unsafe" and r["prediction"] == "safe"]
    #
    #         with open(f"{args.output_file.split('.')[0]}_error_stats.json", 'w') as f:
    #             json.dump({
    #                 "total_examples": len(detailed_results),
    #                 "incorrect_examples": len(incorrect_examples),
    #                 "false_positives": len(false_positives),
    #                 "false_negatives": len(false_negatives),
    #                 "error_rate": len(incorrect_examples) / len(detailed_results),
    #                 "false_positive_rate": len(false_positives) / safe_total if safe_total > 0 else 0,
    #                 "false_negative_rate": len(false_negatives) / unsafe_total if unsafe_total > 0 else 0
    #             }, f, indent=2)
    #         print(f"Error statistics saved to {args.output_file.split('.')[0]}_error_stats.json")


if __name__ == '__main__':
    main()

