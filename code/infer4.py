import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import argparse
import os
import json
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


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


# Function to process dataset for inference (no labels required)
def process_dataset_for_inference(dataset, tokenizer, max_length, batch_size=1000):
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

        return tokenized

    # Process dataset in batches to avoid memory issues
    total_samples = len(dataset)
    logger.info(f"Processing dataset with {total_samples} samples in batches of {batch_size}")

    # Use batched processing with progress bar
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        #batch_size=batch_size,
        remove_columns=dataset.column_names,
        num_proc=16,  # Adjust based on available CPUs
        desc="Processing dataset"
    )

    return tokenized_dataset


# Function to process dataset for inference (no labels required)
def process_dataset_for_evaluation(dataset, tokenizer, max_length, batch_size=1000):
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

        return tokenized
        
    # Process dataset in batches to avoid memory issues
    total_samples = len(dataset)
    logger.info(f"Processing dataset with {total_samples} samples in batches of {batch_size}")

    # Use batched processing with progress bar
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        #batch_size=batch_size,
        remove_columns=dataset.column_names,
        num_proc=16,  # Adjust based on available CPUs
        desc="Processing dataset"
    )
        
    return tokenized_dataset


# Function to run inference without evaluation (no labels)
def run_inference_only(model, dataset, tokenizer, args):
    logger.info("Running inference only (no evaluation)...")

    # Process dataset for inference
    processed_dataset = process_dataset_for_inference(
        dataset, tokenizer, args.max_seq_length
    )

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Run inference with batches
    all_probabilities = []
    all_predictions = []

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        processed_dataset,
        shuffle=False
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            # Move batch to device
            batch = {k: torch.tensor(v).to(device) for k, v in batch.items()}

            # Get model outputs
            outputs = model(**batch)
            logits = outputs.logits

            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(logits, dim=1)

            # Apply threshold
            preds = (probs[:, 1] >= args.threshold).int()

            all_probabilities.append(probs.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())

    # Concatenate all batches
    all_probabilities = np.vstack(all_probabilities)
    all_predictions = np.concatenate(all_predictions)

    # Create detailed results
    logger.info("Processing results...")
    detailed_results = []
    for i in range(len(all_predictions)):
        if all_predictions[i] == 1:
            detailed_results.append({
                "prediction": "unsafe" if all_predictions[i] == 1 else "safe",
                "probabilities": {
                    "safe": float(all_probabilities[i][0]),
                    "unsafe": float(all_probabilities[i][1])
                }
            })

    # Return results
    results = {
        "probabilities": all_probabilities,
        "predictions": all_predictions,
        "detailed_results": detailed_results
    }

    return results


def compute_metrics_with_threshold(eval_pred, threshold=0.7, save_debug_data=False):
    logits, labels = eval_pred

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()

    # Apply custom threshold for predictions
    predictions = np.where(probabilities[:, 1] >= threshold, 1, 0)

    # Optionally save debug data
    if save_debug_data:
        with open('test_eval_data.pkl', 'wb') as fp:
            import pickle
            pickle.dump((probabilities, predictions, labels), fp)
            logger.info("Debug data saved to test_eval_data.pkl")

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=None, labels=[0, 1], zero_division=0
    )
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
                        choices=["train", "validation", "test", "eval"],
                        help='Dataset split to evaluate (default: test)')
    parser.add_argument('--output_file', type=str, help='Path to save detailed results')
    parser.add_argument('--max_seq_length', type=int, default=8192, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of samples to use (default: all available)')
    parser.add_argument('--save_debug_data', action='store_true',
                        help='Save intermediate prediction data for debugging')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to save log output')
    parser.add_argument('--no_eval', action='store_true',
                        help='Run inference only without evaluation (use when no labels are available)')

    args = parser.parse_args()

    # Set up file logging if specified
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Tokenizer loaded. Pad token: {tokenizer.pad_token}")

        # Load model
        logger.info(f"Loading model from {args.model_path}...")
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.to(device)
        logger.info("Model loaded successfully")

        # Load dataset
        logger.info(f"Loading dataset: data/coin, split: {args.dataset_split}")
        try:
            raw_datasets = load_dataset("data/coin", "caller")
            logger.info(
                f"Dataset loaded with {len(raw_datasets[args.dataset_split])} examples in {args.dataset_split} split")

            # Select dataset split and sample if needed
            eval_dataset = raw_datasets[args.dataset_split]

            eval_dataset = eval_dataset.select(range(2000))
            logger.info(f"Using {args.sample_size} samples from {args.dataset_split} split")

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return

        # Check if dataset has labels
        has_labels = False
        if 'label' in eval_dataset.features:
            has_labels = True
            logger.info("Dataset has 'label' field")
        elif 'unsafe' in eval_dataset.features:
            has_labels = True
            logger.info("Dataset has 'unsafe' field that will be used as labels")
        else:
            logger.info("No label field found in dataset, running inference only")
            args.no_eval = True

        # If running without evaluation
        if args.no_eval:
            results = run_inference_only(model, eval_dataset, tokenizer, args)

            # Save results
            if args.output_file:
                output_dir = os.path.dirname(args.output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                with open(args.output_file, 'w') as f:
                    json.dump({
                        "inference_results": results["detailed_results"]
                    }, f, indent=2)
                logger.info(f"Inference results saved to {args.output_file}")

                # Save full predictions if requested
                if args.save_debug_data:
                    full_output_file = f"{os.path.splitext(args.output_file)[0]}_full.pkl"
                    with open(full_output_file, 'wb') as f:
                        import pickle
                        pickle.dump(results, f)
                    logger.info(f"Full results saved to {full_output_file}")

            return

        # If running with evaluation
        # Process dataset
        logger.info("Processing dataset for evaluation...")
        processed_dataset = process_dataset_for_evaluation(
            eval_dataset, tokenizer, args.max_seq_length
        )

        # Create training args for inference
        inference_args = TrainingArguments(
            output_dir="outputs_inference",
            per_device_eval_batch_size=args.batch_size,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            report_to="none",
            dataloader_num_workers=4,  # Parallel data loading
        )

        # Create trainer for inference
        trainer = Trainer(
            model=model,
            args=inference_args,
            compute_metrics=lambda pred: compute_metrics_with_threshold(
                pred, threshold=args.threshold, save_debug_data=args.save_debug_data
            ),
        )

        # Run prediction
        logger.info(f"Running inference with threshold: {args.threshold}...")
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
        logger.info("Processing detailed results...")
        detailed_results = []
        for i in tqdm(range(len(thresholded_predictions)), desc="Creating detailed results"):
            detailed_results.append({
                "prediction": "unsafe" if thresholded_predictions[i] == 1 else "safe",
                "ground_truth": "unsafe" if labels[i] == 1 else "safe",
                "probabilities": {
                    "safe": float(probabilities[i][0]),
                    "unsafe": float(probabilities[i][1])
                },
                "correct": (thresholded_predictions[i] == labels[i])
            })

        # Print metrics
        logger.info("\nEvaluation Results with Custom Threshold:")
        logger.info(f"Threshold used: {args.threshold}")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                logger.info(f"{metric_name}: {metric_value:.4f}")
            else:
                logger.info(f"{metric_name}: {metric_value}")

        # Calculate error analysis statistics
        safe_total = metrics['test_true_negatives'] + metrics['test_false_positives']
        unsafe_total = metrics['test_true_positives'] + metrics['test_false_negatives']
        logger.info("\nError Analysis:")
        logger.info(
            f"Safe functions: {safe_total} (Correctly classified: {metrics['test_true_negatives']} [{metrics['test_true_negatives'] / safe_total * 100:.2f}%])")
        logger.info(
            f"Unsafe functions: {unsafe_total} (Correctly classified: {metrics['test_true_positives']} [{metrics['test_true_positives'] / unsafe_total * 100:.2f}%])")

        # Save detailed results if specified
        if args.output_file:
            output_dir = os.path.dirname(args.output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(args.output_file, 'w') as f:
                json.dump({
                    "metrics": {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in
                                metrics.items()},
                    "detailed_results": detailed_results[:1000]  # Limit to 1000 entries
                }, f, indent=2)
            logger.info(f"Detailed results saved to {args.output_file}")

            # Save stats about incorrectly classified examples
            if len(detailed_results) > 1000:
                incorrect_examples = [r for r in detailed_results if not r["correct"]]
                false_positives = [r for r in detailed_results if
                                   r["ground_truth"] == "safe" and r["prediction"] == "unsafe"]
                false_negatives = [r for r in detailed_results if
                                   r["ground_truth"] == "unsafe" and r["prediction"] == "safe"]

                error_stats_file = f"{os.path.splitext(args.output_file)[0]}_error_stats.json"
                with open(error_stats_file, 'w') as f:
                    json.dump({
                        "total_examples": len(detailed_results),
                        "incorrect_examples": len(incorrect_examples),
                        "false_positives": len(false_positives),
                        "false_negatives": len(false_negatives),
                        "error_rate": len(incorrect_examples) / len(detailed_results),
                        "false_positive_rate": len(false_positives) / safe_total if safe_total > 0 else 0,
                        "false_negative_rate": len(false_negatives) / unsafe_total if unsafe_total > 0 else 0
                    }, f, indent=2)
                logger.info(f"Error statistics saved to {error_stats_file}")

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    main()

