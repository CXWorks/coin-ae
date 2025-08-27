import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets
import numpy as np
from transformers import DataCollatorForLanguageModeling
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os

os.environ["WANDB_DISABLED"] = "true"

# Configuration parameters
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
load_in_4bit = True
max_seq_length = 8192  # Significantly reduced from 8192
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LLMWithClassificationHead(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size

        # Get the dtype and device of the base model
        base_param = next(base_model.parameters())
        base_dtype = base_param.dtype
        base_device = base_param.device

        # Initialize classifier with the same dtype and device as the base model
        self.classifier = nn.Linear(hidden_size, 2)  # Binary classification

        # Convert classifier weights to match base model dtype and device
        self.classifier.weight.data = self.classifier.weight.data.to(device=base_device, dtype=base_dtype)
        self.classifier.bias.data = self.classifier.bias.data.to(device=base_device, dtype=base_dtype)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Get the base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Get the last hidden state
        hidden_states = outputs.hidden_states[-1]

        # Find the position of the last non-padding token
        if attention_mask is not None:
            last_token_positions = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]

            # Gather the hidden states of the last tokens
            last_token_hidden_states = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                last_token_positions
            ]
        else:
            # If no attention mask, just use the last token
            last_token_hidden_states = hidden_states[:, -1]

        # Ensure classifier parameters are on the same device and have the same dtype as the inputs
        if (self.classifier.weight.device != last_token_hidden_states.device or
                self.classifier.weight.dtype != last_token_hidden_states.dtype):
            self.classifier = self.classifier.to(
                device=last_token_hidden_states.device,
                dtype=last_token_hidden_states.dtype
            )

        # Get logits from the classifier
        logits = self.classifier(last_token_hidden_states)

        # Make sure labels are on the correct device if provided
        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            # Ensure labels are on the same device as logits
            labels = labels.to(logits.device)
            loss = criterion(logits, labels.view(-1))

        return {"loss": loss, "logits": logits}


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Convert logits to predictions
    predictions = np.argmax(logits, axis=1)

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


# Custom data collator for classification
class ClassificationDataCollator:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        # Tokenize the texts
        batch = {}

        # Extract input_ids and attention_mask
        batch["input_ids"] = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        batch["attention_mask"] = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)

        # Extract labels if available (for training)
        if "label" in features[0]:
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long)

        return batch


# Function to format prompts
def formatting_prompts_func(example):
    prompt = """Here is a Rust code and please check if the function starting with `>` is safe or unsafe:
    {}

    Is this function unsafe? Answer with "Yes" or "No".

    SOLUTION
    The correct answer is: "{}" """

    if isinstance(example['function_text'], str):
        label = "Yes" if example['label'] == 1 else "No"
        return prompt.format(example['function_text'], label)

    texts = []
    for i in range(len(example['function_text'])):
        label = "Yes" if example['label'][i] == 1 else "No"
        texts.append(prompt.format(example['function_text'][i], label))
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
            tokenized["label"] = examples['label']
        else:
            tokenized["label"] = examples['label']

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

    # Load model and apply optimizations
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        load_in_4bit=load_in_4bit,
        device_map="auto"
    )

    # Enable gradient checkpointing to save memory
    # base_model.gradient_checkpointing_enable()

    # Apply LoRA for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
    )

    base_model = get_peft_model(base_model, lora_config)

    # Create the model with classification head
    model = LLMWithClassificationHead(base_model)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} ({trainable_params / all_params:.2%} of all parameters)")

    # Load and prepare datasets
    raw_datasets = load_dataset("data/coin", "caller")

    # Balance the training dataset
    train_dataset = raw_datasets['train']
    label_0_data = train_dataset.filter(lambda x: x['label'] == 0)
    label_1_data = train_dataset.filter(lambda x: x['label'] == 1)
    print(f'0 :{len(label_0_data)} 1: {len(label_1_data)}')
    num_label_1 = len(label_1_data)
    indices = np.random.choice(len(label_0_data), num_label_1, replace=False)
    label_0_random_sample = label_0_data.select(indices)

    balanced_train_dataset = concatenate_datasets([label_0_random_sample, label_1_data]).shuffle()
    raw_datasets['train'] = balanced_train_dataset

    # Resize validation dataset to prevent OOM
    raw_datasets["validation"] = raw_datasets["validation"].shuffle().select(range(40000))

    # Process datasets for classification
    train_dataset = process_dataset_for_classification(raw_datasets['train'], tokenizer, max_seq_length)
    eval_dataset = process_dataset_for_classification(raw_datasets['validation'], tokenizer, max_seq_length)
    test_dataset = process_dataset_for_classification(raw_datasets['test'], tokenizer, max_seq_length)

    # Create data collator
    data_collator = ClassificationDataCollator(tokenizer, max_seq_length)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="outputs_8b_classifier",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        learning_rate=1e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        # Force all tensors to use the same precision
        # torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        metric_for_best_model="eval_precision_unsafe",
        load_best_model_at_end=True,
        num_train_epochs=3,
        report_to="none",
        group_by_length=True,
        evaluation_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=3,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the trained model
    trainer.save_model("final_model_classifier.pt")

    # Evaluate on test set
    print('Starting final evaluation...')
    model.eval()
    results = trainer.evaluate(eval_dataset=test_dataset)

    # Print results
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

