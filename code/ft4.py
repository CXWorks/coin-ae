# needed as this function doesn't like it when the lm_head has its size changed
# from unsloth import tokenizer_utils
# def do_nothing(*args, **kwargs):
#     pass
# tokenizer_utils.fix_untrained_tokens = do_nothing

import torch
from torch.nn import Module
from collections import OrderedDict, defaultdict
from typing import Mapping, Any, List, NamedTuple
major_version, minor_version = torch.cuda.get_device_capability()
print(f"Major: {major_version}, Minor: {minor_version}")
from datasets import load_dataset
import datasets
from datasets import concatenate_datasets
from trl import SFTTrainer
import pandas as pd
import numpy as np
import os
import pandas as pd
import numpy as np
# from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer
from trl import SFTTrainer
from transformers import TrainingArguments, Trainer
from typing import Tuple
import warnings
from typing import Any, Dict, List, Union
from transformers import DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
os.environ["WANDB_DISABLED"] = "true"
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
max_seq_length =  8192# Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+


model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B";load_in_4bit = True
# model_name = "unsloth/Llama-3.2-3B-bnb-4bit";load_in_4bit = True,

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np


def get_predictions(logits, labels):
    # Get predictions from logits
    # predictions =

    # Filter out ignored labels (-100)
    valid_indices = labels != -100
    valid_predictions = np.argmax(logits[valid_indices], axis=-1)
    valid_labels = labels[valid_indices]

    return valid_predictions, valid_labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Get predictions for the last token only (since that's what we trained on)
    # last_token_logits = logits[:, -1, :] if len(logits.shape) > 2 else logits

    # Get predictions and valid labels
    predictions, true_labels = get_predictions(logits, labels)
    # print(predictions)
    # print(true_labels)
    # import pickle
    # with open('metrc.pkl', 'wb') as f:
    #     pickle.dump((logits, labels, predictions, true_labels), f)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, labels = [0, 1], zero_division=0, average=None)
    conf_matrix = confusion_matrix(true_labels, predictions, labels = [0, 1])

    # Calculate per-class metrics
    results = {
        'accuracy': accuracy,
        'precision_safe': precision[0],  # No class
        'precision_unsafe': precision[1],  # Yes class
        'recall_safe': recall[0],  # No class
        'recall_unsafe': recall[1],  # Yes class
        'f1_safe': f1[0],  # No class
        'f1_unsafe': f1[1],  # Yes class
    }

    # Add confusion matrix elements
    results.update({
        'true_negatives': conf_matrix[0][0],
        'false_positives': conf_matrix[0][1],
        'false_negatives': conf_matrix[1][0],
        'true_positives': conf_matrix[1][1]
    })

    # Calculate macro and weighted averages
    results.update({
        'macro_precision': np.mean(precision),
        'macro_recall': np.mean(recall),
        'macro_f1': np.mean(f1),
        # Weighted metrics based on class distribution
        'weighted_precision': np.average(precision, weights=[np.sum(true_labels == 0), np.sum(true_labels == 1)]),
        'weighted_recall': np.average(recall, weights=[np.sum(true_labels == 0), np.sum(true_labels == 1)]),
        'weighted_f1': np.average(f1, weights=[np.sum(true_labels == 0), np.sum(true_labels == 1)])
    })

    return results
from peft import LoraConfig, get_peft_model

# def load_model(checkpoint_path="outputs_8b/checkpoint-3500",
#                base_model="unsloth/Llama-3.2-3B-bnb-4bit"):
#     model, tokenizer = FastLanguageModel.from_pretrained(checkpoint_path)
#     return model, tokenizer


if __name__ == '__main__':

    model = AutoModelForCausalLM.from_pretrained(
        # pretrained_model_name_or_path="outputs/checkpoint-1700",
        pretrained_model_name_or_path = model_name ,
        load_in_4bit = load_in_4bit,
        # max_seq_length = max_seq_length,
        # dtype = dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_name)
    # model, tokenizer = load_model()


    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
    print('Yes no', yes_token_id, no_token_id)
    # keep only the yes and no tokens from lm_head
    par = torch.nn.Parameter(torch.vstack([model.lm_head.weight[no_token_id, :], model.lm_head.weight[yes_token_id, :]]))
    print(par.shape)
    print(model.lm_head.weight.shape)
    model.lm_head.weight = par

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "lm_head",  # can easily be trained because it has only 2 tokens
            "q_proj", "k_proj", "v_proj", "o_proj"],
            # "gate_proj", "up_proj", "down_proj", ],
        lora_dropout=0.1,
        bias="lora_only",
    )

    model = get_peft_model(model, config)


    model.print_trainable_parameters()


    raw_datasets = load_dataset(
            "data/coin",
            "caller"
    )
    train_dataset = raw_datasets['train']

    # Split the training dataset based on labels
    label_0_data = train_dataset.filter(lambda x: x['label'] == 0)
    label_1_data = train_dataset.filter(lambda x: x['label'] == 1)

    # Calculate the number of samples in label 1
    num_label_1 = len(label_1_data)

    # Randomly select samples from label 0 to match the number of label 1 samples
    indices = np.random.choice(len(label_0_data), num_label_1, replace=False)
    label_0_random_sample = label_0_data.select(indices)

    # Concatenate the sampled label 0 data with label 1 data
    balanced_train_dataset = concatenate_datasets([label_0_random_sample, label_1_data]).shuffle()

    # Update the training dataset in the raw_datasets
    raw_datasets['train'] = balanced_train_dataset

    # raw_datasets["train"] = raw_datasets["train"].shuffle()
    raw_datasets["validation"] = raw_datasets["validation"].shuffle().select(range(160000))
    prompt = """Here is a Rust code and please check if the function starting with `>` is safe or unsafe:
    {}

    Is this function unsafe? Answer with "Yes" or "No".

    SOLUTION
    The correct answer is: "{}"""
    positivelabel = "Yes"
    negativelabel = "No"

    def formatting_prompts_func(dataset_):

        if isinstance(dataset_['function_text'], str):
            label = positivelabel if dataset_['label'] == 1 else negativelabel
            text = prompt.format(dataset_['function_text'], label)
            # print('*'*20)
            # print(text)
            return text
        texts = []
        for i in range(len(dataset_['function_text'])):
            t = dataset_['function_text'][i]
        #    print(dataset_)
            label = positivelabel if dataset_['label'][i] == 1 else negativelabel
            text = prompt.format(t, label)
            texts.append(text)
            # print('='*20)
            # print(text)
        return texts

    # this custom collator is needed to change the sequence labels from yes_token_id and no_token_id to 1 and 0. It also trains only on the last token of the sequence.
    class DataCollatorForLastTokenLM(DataCollatorForLanguageModeling):
        def __init__(
                self,
                *args,
                mlm: bool = False,
                ignore_index: int = -100,
                class_weights: torch.Tensor = None,
                **kwargs,
        ):
            super().__init__(*args, mlm=mlm, **kwargs)
            self.ignore_index = ignore_index
            self.class_weights = class_weights

        def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
            batch = super().torch_call(examples)

            for i in range(len(examples)):
                # Find the last non-padding token
                last_token_idx = (batch["labels"][i] != self.ignore_index).nonzero()[-1].item()
                # print(f"Example {i} transformed before label: {batch['labels'][i]} {batch['labels'][i][last_token_idx]}")
                # print(tokenizer.decode(batch['labels'][i]))
                # Set all labels to ignore_index except for the last token
                batch["labels"][i, :last_token_idx] = self.ignore_index
                # The old labels for the Yes and No tokens need to be mapped to 1 and 0
                batch["labels"][i, last_token_idx] = 1 if batch["labels"][i, last_token_idx] == yes_token_id else 0
                # Debugging statement for transformed labels
                # print(f"Example {i} transformed label: {batch['labels'][i][last_token_idx]}")

                # Debugging statement to see batch labels
            # print("Batch Labels after transformation:", batch["labels"])

            return batch

    # Calculate class weights
    train_labels = np.array(raw_datasets['train']['label'])
    class_weights = torch.tensor(len(train_labels) / (2 * np.bincount(train_labels)), dtype=torch.float32)
    collator = DataCollatorForLastTokenLM(tokenizer=tokenizer, class_weights=class_weights)
    print(class_weights)


    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=raw_datasets['train'],
        eval_dataset=raw_datasets['validation'],
        max_seq_length=max_seq_length,
        dataset_num_proc=64,
        packing=False,  # not needed because group_by_length is True
        compute_metrics=compute_metrics,

        args=TrainingArguments(
            per_device_train_batch_size=16,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=100,
            learning_rate=1e-7,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir="outputs_8b_test",
            metric_for_best_model = "eval_precision_unsafe",
            load_best_model_at_end=True,
            # max_steps=5000,
            # report_to = "wandb",
            num_train_epochs=5,
            report_to="none",
            group_by_length=True,
            eval_strategy="steps",
            eval_steps=3500,
            save_strategy="steps",
            save_steps=3500,
            save_total_limit=5,

        ),
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    trainer_stats = trainer.train()
    trainer.save_model("final_model2.pt")
    print('start eval!!!!!!!!')



    # Run evaluation
    model.eval()
    results = trainer.evaluate(eval_dataset=raw_datasets['test'])

    # Print results
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    # model.eval()
