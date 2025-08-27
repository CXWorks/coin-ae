import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, Qwen2ForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, concatenate_datasets
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os

# Configuration parameters
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
max_seq_length = 8192
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    import pickle
    # Convert logits to predictions
    predictions = np.argmax(logits, axis=1)
    with open('qwen_data.pkl', 'wb') as fp:
        pickle.dump((logits, labels), fp)
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
    }

    return results


# Function to format prompts
def formatting_prompts_func(example):
    prompt = """Here is a Rust code and please check if the function starting with `>` is safe or unsafe:
    {}

    Is this function unsafe?
    """

    if isinstance(example['function_text'], str):
        # label = "Yes" if example['label'] == 1 else "No"
        return prompt.format(example['function_text'])

    texts = []
    for i in range(len(example['function_text'])):
        # label = "Yes" if example['label'][i] == 1 else "No"
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
        if isinstance(examples['label'], int):
            tokenized["labels"] = examples['label']  # Note: using "labels" instead of "label"
        else:
            tokenized["labels"] = examples['label']  # Note: using "labels" instead of "label"

        return tokenized

    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=64
    )

    return tokenized_dataset


if __name__ == '__main__':
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path='./final_model_classifier.pt/', #model_name,
        num_labels=2,  # Binary classification
        id2label={0: 'safe', 1: 'unsafe'},
        label2id={'safe': 0, 'unsafe': 1}
    )


    # Enable gradient checkpointing to save memory before applying LoRA
    #model.gradient_checkpointing_enable()

    # Move to train mode before applying LoRA
    model.train()

    # Apply LoRA for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            # "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS  # Changed to sequence classification task type
    )

    model = get_peft_model(model, lora_config)

    # Print model info (optional)
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    # Verify trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable: {name}, {param.shape}")

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} ({trainable_params / all_params:.2%} of all parameters)")

    # Load and prepare datasets
    raw_datasets = load_dataset("data/coin", "caller")
    raw_d2 = load_dataset("data/coinstd", "caller")
    # Balance the training dataset
    train_dataset = raw_datasets['train']
    label_0_data = train_dataset.filter(lambda x: x['label'] == 0)
    label_1_data = train_dataset.filter(lambda x: x['label'] == 1)
    print(f'0: {len(label_0_data)} 1: {len(label_1_data)}')
    num_label_1 = len(label_1_data)
    indices = np.random.choice(len(label_0_data), num_label_1, replace=False)
    label_0_random_sample = label_0_data.select(indices)

    balanced_train_dataset = concatenate_datasets([label_0_random_sample, label_1_data]).shuffle()
    raw_datasets['train'] = balanced_train_dataset

    # Resize validation dataset to prevent OOM
    raw_datasets["validation"] = raw_datasets["validation"].shuffle().select(range(20000))

    # Process datasets for classification
    train_dataset = process_dataset_for_classification(raw_datasets['train'], tokenizer, max_seq_length)
    eval_dataset = process_dataset_for_classification(raw_datasets['validation'], tokenizer, max_seq_length)
    test_dataset = process_dataset_for_classification(raw_d2['test'].shuffle(), tokenizer, max_seq_length)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="outputs_8b_classifier_sp",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        learning_rate=1e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        metric_for_best_model="eval_precision_unsafe",
        load_best_model_at_end=True,
        num_train_epochs=5,
        report_to="none",
        group_by_length=True,
        evaluation_strategy="steps",
        eval_steps=1100,
        save_strategy="steps",
        save_steps=1100,
        save_total_limit=3,
    )

    # Create trainer (no need for custom data collator)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    # trainer.train()

    # Save the trained model
    # trainer.save_model("final_model_classifier.pt")

    # Evaluate on test set
    print('Starting final evaluation...')
    model.eval()
    results = trainer.evaluate(eval_dataset=test_dataset)

    # Print results
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
