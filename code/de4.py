from typing import List, Union, Any, Dict

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets
import numpy as np
from transformers import DataCollatorForLanguageModeling
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.dataloader import DataLoader
import argparse
import pickle
os.environ["WANDB_DISABLED"] = "true"
from datetime import datetime, timedelta
# Configuration parameters
# model_name = "meta-llama/Llama-3.2-3B"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
load_in_4bit = True
max_seq_length = 8192
os.environ['NCCL_TIMEOUT'] = '3600'
os.environ['NCCL_DEBUG'] = 'INFO'  # or 'WARN'

def setup(rank, world_size):
    """
    Initialize the distributed environment
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    print('init', rank, world_size)
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(hours=2))
    print('finish init')
    # Set device for this process
    torch.cuda.set_device(rank)


def cleanup():
    """
    Clean up the distributed environment
    """
    dist.destroy_process_group()


class LLMWithClassificationHead(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size

        # Get the dtype of the base model
        base_dtype = next(base_model.parameters()).dtype

        # Initialize classifier with the same dtype as the base model
        self.classifier = nn.Linear(hidden_size, 2)  # Binary classification

        # Convert classifier weights to match base model dtype
        self.classifier.weight.data = self.classifier.weight.data.to(base_dtype)
        self.classifier.bias.data = self.classifier.bias.data.to(base_dtype)

        # Explicitly set requires_grad=True for classifier parameters
        self.classifier.weight.requires_grad = True
        self.classifier.bias.requires_grad = True

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

            # Make sure last_token_positions doesn't exceed sequence length
            last_token_positions = torch.clamp(last_token_positions, 0, hidden_states.size(1) - 1)

            # Gather the hidden states of the last tokens
            last_token_hidden_states = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                last_token_positions
            ]
        else:
            # If no attention mask, just use the last token
            last_token_hidden_states = hidden_states[:, -1]

        # Make sure the classifier has the same dtype as the hidden states
        if self.classifier.weight.dtype != last_token_hidden_states.dtype:
            self.classifier.weight.data = self.classifier.weight.data.to(last_token_hidden_states.dtype)
            self.classifier.bias.data = self.classifier.bias.data.to(last_token_hidden_states.dtype)

        # Get logits from the classifier
        logits = self.classifier(last_token_hidden_states)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
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
            padding_side = "right",
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
        num_proc=16
    )

    return tokenized_dataset


def get_indices_for_balancing(dataset, seed):
    """Get indices for balanced dataset with fixed seed"""
    label_0_data = dataset.filter(lambda x: x['label'] == 0)
    label_1_data = dataset.filter(lambda x: x['label'] == 1)
    num_label_1 = len(label_1_data)

    # Use fixed seed for deterministic sampling
    rng = np.random.RandomState(seed)
    sampled_indices = rng.choice(len(label_0_data), num_label_1, replace=False)

    # Get actual indices from the original dataset
    indices_0 = [i for i, x in enumerate(dataset) if x['label'] == 0]
    indices_1 = [i for i, x in enumerate(dataset) if x['label'] == 1]

    selected_indices_0 = [indices_0[i] for i in sampled_indices]
    all_indices = selected_indices_0 + indices_1

    # Shuffle with fixed seed
    rng.shuffle(all_indices)
    return all_indices


def train(rank, world_size, args):
    # Setup the process group
    setup(rank, world_size)

    # Set the device
    device = torch.device(f"cuda:{rank}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    raw_datasets = load_dataset("data/coin", "caller")

    # SYNC POINT 1: Balance datasets deterministically
    # Only rank 0 computes indices, but all processes will use them
    if rank == 0 and False:
        print("Process 0: Computing balanced dataset indices...")
        train_indices = get_indices_for_balancing(raw_datasets['train'], args.seed)
        val_indices = list(range(min(20000, len(raw_datasets['validation']))))

        # Save indices to file
        with open('/mnt/sdc/xiang/cache/train_indices.pkl', 'wb') as f:
            pickle.dump(train_indices, f)
        with open('/mnt/sdc/xiang/cache/val_indices.pkl', 'wb') as f:
            pickle.dump(val_indices, f)

    # Make sure rank 0 has finished computing indices
    #dist.barrier()

    # All processes load the same indices
    with open('/mnt/sdc/xiang/cache/train_indices.pkl', 'rb') as f:
        train_indices = pickle.load(f)
    with open('/mnt/sdc/xiang/cache/val_indices.pkl', 'rb') as f:
        val_indices = pickle.load(f)

    # All processes select the same examples
    balanced_train_dataset = raw_datasets['train'].select(train_indices)
    validation_dataset = raw_datasets['validation'].select(val_indices)

    if rank == 0 :
        # Verify data balance
        label_counts = {}
        for label in balanced_train_dataset['label']:
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"Balanced dataset class distribution: {label_counts}")

        # SYNC POINT 2: Process datasets with tokenizer
    train_dataset = process_dataset_for_classification(balanced_train_dataset, tokenizer, max_seq_length)
    eval_dataset = process_dataset_for_classification(validation_dataset, tokenizer, max_seq_length)
    test_dataset = process_dataset_for_classification(raw_datasets['test'], tokenizer, max_seq_length)

    # Verify all processes have the same data size
    sizes = [len(train_dataset), len(eval_dataset), len(test_dataset)]
    if rank == 0:
        print(f"Process {rank}: Dataset sizes - Train: {sizes[0]}, Val: {sizes[1]}, Test: {sizes[2]}")

    # Load model and apply optimizations
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        #load_in_8bit=load_in_4bit,
    )

    # Enable gradient checkpointing to save memory before applying LoRA
    base_model.gradient_checkpointing_enable()

    # Move to train mode before applying LoRA
    base_model.train()

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
        task_type=TaskType.CAUSAL_LM  # Explicitly specify task type
    )

    base_model = get_peft_model(base_model, lora_config)

    # Create the model with classification head
    model = LLMWithClassificationHead(base_model)
    model.to(device)    

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    # model.to(device)
    if rank == 0:
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params} ({trainable_params / all_params:.2%} of all parameters)")


    # Make sure all processes see the same datasets
    torch.distributed.barrier()

    print('=' * 20, '>>', rank, len(train_dataset), len(eval_dataset), len(test_dataset))

    # Process datasets for classification
    # train_dataset = process_dataset_for_classification(raw_datasets['train'], tokenizer, max_seq_length)
    # eval_dataset = process_dataset_for_classification(raw_datasets['validation'], tokenizer, max_seq_length)
    # test_dataset = process_dataset_for_classification(raw_datasets['test'], tokenizer, max_seq_length)

    # Create distributed sampler - crucial for proper data distribution
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        seed=args.seed
    )

    # Create data collator
    data_collator = ClassificationDataCollator(tokenizer = tokenizer)

    # Define training arguments with distributed settings
    training_args = TrainingArguments(
        output_dir=f"outputs_ddp",
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
        seed=args.seed,
        metric_for_best_model="eval_precision_unsafe",
        load_best_model_at_end=True,
        num_train_epochs=10,
        report_to="none" if rank == 0 else "none",
        group_by_length=True,
        evaluation_strategy="steps",
        eval_steps=11000,
        save_strategy="steps",
        save_steps=11000,
        save_total_limit=3,
        local_rank=rank,
        ddp_find_unused_parameters=True,
        remove_unused_columns=False
    )
    print(train_dataset)
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics if rank == 0 else None,
    )

    # Important: Set the sampler for the dataloader
    trainer._get_train_sampler = lambda: train_sampler

    torch.distributed.barrier()

    # Train the model
    trainer.train()

    # Save the trained model (only on rank 0)
    if rank == 0:
        # Get the unwrapped model
        unwrapped_model = model.module
        # Save the model
        trainer.save_model(f"final_model_classifier_ddp.pt")

        # Evaluate on test set
        print('Starting final evaluation...')
        unwrapped_model.eval()
        results = trainer.evaluate(eval_dataset=test_dataset)

        # Print results
        print("\nEvaluation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

    # Clean up
    cleanup()


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Distributed training script')
    parser.add_argument('--world-size', type=int, default=torch.cuda.device_count(),
                        help='number of GPUs to use for training')
    parser.add_argument('--seed', type=int, default=3407,
                        help='random seed for reproducibility')
    parser.add_argument('--local-rank', type=int, default=3407,
                        help='random seed for reproducibility')
    args = parser.parse_args()

    # Print total available GPUs
    print(f"Training with {args.world_size} GPUs")
    #local_rank = int(os.environ['LOCAL_RANK'])
    #train(local_rank, args.world_size, args)
    # Use all available GPUs
    mp.spawn(train, args=(args.world_size, args), nprocs=args.world_size, join=True)

