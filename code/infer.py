# needed as this function doesn't like it when the lm_head has its size changed
from unsloth import tokenizer_utils
def do_nothing(*args, **kwargs):
    pass
tokenizer_utils.fix_untrained_tokens = do_nothing

import torch
major_version, minor_version = torch.cuda.get_device_capability()
print(f"Major: {major_version}, Minor: {minor_version}")
from datasets import load_dataset
import datasets
from trl import SFTTrainer
import pandas as pd
import numpy as np
import os
import pandas as pd
import numpy as np
from unsloth import FastLanguageModel
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
from peft import PeftModel

# model_name = "unsloth/Qwen2-7B-bnb-4bit";load_in_4bit = True
model_name = "unsloth/Llama-3.2-3B-bnb-4bit";load_in_4bit = True,

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

import torch
from torch.nn import Module
from collections import OrderedDict, defaultdict
from typing import Mapping, Any, List, NamedTuple

from unsloth import tokenizer_utils


def do_nothing(*args, **kwargs):
    pass


tokenizer_utils.fix_untrained_tokens = do_nothing

from datasets import load_dataset
import datasets
from trl import SFTTrainer
import pandas as pd
import numpy as np
import os
import pandas as pd
import numpy as np
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, Trainer
from typing import Tuple
import warnings
from typing import Any, Dict, List, Union
from transformers import DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer


def _find_mismatched_keys(
        model: torch.nn.Module, peft_model_state_dict: dict[str, torch.Tensor], ignore_mismatched_sizes: bool = True
) -> tuple[dict[str, torch.Tensor], list[tuple[str, tuple[int, ...], tuple[int, ...]]]]:
    return peft_model_state_dict, []


# Monkey patch the original function
import peft.utils.save_and_load

peft.utils.save_and_load._find_mismatched_keys = _find_mismatched_keys


class _IncompatibleKeys(NamedTuple):
    missing_keys: List[str]
    unexpected_keys: List[str]


def patched_load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Expected state_dict to be dict-like, got {type(state_dict)}.")

    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    error_msgs: List[str] = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = OrderedDict(state_dict)
    if metadata is not None:
        state_dict._metadata = metadata  # type: ignore[attr-defined]

    def load(module, local_state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        if assign:
            local_metadata["assign_to_params_buffers"] = assign
        module._load_from_state_dict(
            local_state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + "."
                child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                load(child, child_state_dict, child_prefix)

        incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
        for hook in module._load_state_dict_post_hooks.values():
            out = hook(module, incompatible_keys)
            assert out is None, (
                "Hooks registered with ``register_load_state_dict_post_hook`` are not"
                "expected to return new values, if incompatible_keys need to be modified,"
                "it should be done inplace."
            )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        for name, param in self._parameters.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if param.shape != input_param.shape:
                    print(
                        f"Shape mismatch for {key}, creating new tensor. Old shape: {param.shape}, New shape: {input_param.shape}")
                    # Create a new parameter with the shape from state_dict
                    new_param = torch.nn.Parameter(torch.empty_like(input_param), requires_grad=param.requires_grad)
                    new_param.data.copy_(input_param)
                    setattr(self, name, new_param)
                else:
                    param.data.copy_(input_param)
            elif strict:
                missing_keys.append(key)

        for name, buf in self._buffers.items():
            key = prefix + name
            if key in state_dict:
                input_buf = state_dict[key]
                if buf.shape != input_buf.shape:
                    print(
                        f"Shape mismatch for buffer {key}, creating new tensor. Old shape: {buf.shape}, New shape: {input_buf.shape}")
                    # Create a new buffer with the shape from state_dict
                    new_buf = torch.empty_like(input_buf)
                    new_buf.copy_(input_buf)
                    setattr(self, name, new_buf)
                else:
                    buf.copy_(input_buf)
            elif strict:
                missing_keys.append(key)

    # Monkey patch the _load_from_state_dict method
    Module._load_from_state_dict = _load_from_state_dict

    load(self, state_dict)
    del load

    if strict:
        if len(unexpected_keys) > 0:
            error_msgs.insert(0, "Unexpected key(s) in state_dict: {}. ".format(
                ", ".join(f'"{k}"' for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(0, "Missing key(s) in state_dict: {}. ".format(", ".join(f'"{k}"' for k in missing_keys)))

    if len(error_msgs) > 0:
        raise RuntimeError(
            "Error(s) in loading state_dict for {}:\n\t{}".format(self.__class__.__name__, "\n\t".join(error_msgs)))

    return _IncompatibleKeys(missing_keys, unexpected_keys)


# Apply the monkey patch
Module.load_state_dict = patched_load_state_dict

# Load model



def get_predictions(logits, labels):
    # Get predictions from logits
    predictions = np.argmax(logits, axis=-1)

    # Filter out ignored labels (-100)
    valid_indices = labels != -100
    valid_predictions = predictions[valid_indices]
    valid_labels = labels[valid_indices]

    return valid_predictions, valid_labels

from collections import Counter
def compute_metrics_with_counts(eval_pred):
    """Compute evaluation metrics with detailed counts"""
    logits, labels = eval_pred

    # Get predictions for the last token
    last_token_logits = logits[:, -1, :] if len(logits.shape) > 2 else logits

    # Get predictions and valid labels
    predictions = np.argmax(last_token_logits, axis=-1)
    valid_indices = labels[:, -1] != -100
    valid_predictions = predictions[valid_indices]
    valid_labels = labels[valid_indices][:, -1]

    # Calculate basic metrics
    accuracy = accuracy_score(valid_labels, valid_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(valid_labels, valid_predictions, average=None)
    conf_matrix = confusion_matrix(valid_labels, valid_predictions)

    # Count actual labels
    label_counts = Counter(valid_labels)

    # Count predictions
    pred_counts = Counter(valid_predictions)

    # Count correct predictions per class
    correct_predictions = valid_predictions == valid_labels
    correct_per_class = {
        0: np.sum((valid_labels == 0) & correct_predictions),
        1: np.sum((valid_labels == 1) & correct_predictions)
    }

    results = {
        # Basic metrics
        'accuracy': accuracy,
        'precision_safe': precision[0],
        'precision_unsafe': precision[1],
        'recall_safe': recall[0],
        'recall_unsafe': recall[1],
        'f1_safe': f1[0],
        'f1_unsafe': f1[1],

        # Confusion matrix
        'true_negatives': conf_matrix[0][0],
        'false_positives': conf_matrix[0][1],
        'false_negatives': conf_matrix[1][0],
        'true_positives': conf_matrix[1][1],

        # Actual label counts
        'actual_no_count': int(label_counts[0]),
        'actual_yes_count': int(label_counts[1]),

        # Predicted label counts
        'predicted_no_count': int(pred_counts[0]),
        'predicted_yes_count': int(pred_counts[1]),

        # Correct predictions per class
        'correct_no_count': int(correct_per_class[0]),
        'correct_yes_count': int(correct_per_class[1])
    }

    # Add aggregate metrics
    results.update({
        'macro_precision': np.mean(precision),
        'macro_recall': np.mean(recall),
        'macro_f1': np.mean(f1),
        'total_samples': len(valid_labels)
    })

    return results



def load_model(checkpoint_path="outputs_sec/checkpoint-3000",
               base_model="unsloth/Llama-3.2-3B-bnb-4bit"):
    model, tokenizer = FastLanguageModel.from_pretrained(checkpoint_path)
    return model, tokenizer


if __name__ == '__main__':

    model, tokenizer = load_model()
    FastLanguageModel.for_inference(model)

    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]



    raw_datasets = load_dataset(
            "data/coin",
            "caller"
    )

    prompt = """Here is a Rust code and please check if the function starting with `>` is safe or unsafe:
    {}

    Is this function unsafe? Answer with "Yes" or "No".

    SOLUTION
    The correct answer is: "{}"""
    positivelabel = "Yes"
    negativelabel = "No"

    def formatting_test_func(dataset_):
        if isinstance(dataset_['function_text'], str):
            label = positivelabel if dataset_['label'] == 1 else negativelabel
            return {'function_text':prompt.format(dataset_['function_text'], label)}
        texts = []
        for i in range(len(dataset_['function_text'])):
            t = dataset_['function_text'][i]
        #    print(dataset_)
        #     label = positivelabel if dataset_['label'][i] == 1 else negativelabel
            text = prompt.format(t, '')
            texts.append(text)
        return texts

    def formatting_prompts_func(dataset_):
        if isinstance(dataset_['function_text'], str):
            return ' '
        texts = []
        for i in range(len(dataset_['function_text'])):
            t = dataset_['function_text'][i]
            label = positivelabel if dataset_['label'][i] == 1 else negativelabel
            text = prompt.format(t, label)
            texts.append(text)
        return texts


    test_dataset = raw_datasets['test'].shuffle().select(range(3200)).map(
        formatting_test_func,
        batched=False,
        num_proc=64,
        desc="Running format on every text in test dataset",
    )
    print(test_dataset)
    def tokenize_function(examples):
        return tokenizer(examples['function_text'], max_length=max_seq_length)
    tokenized_test_dataset = test_dataset.map(
        tokenize_function,
        batched=False,
        num_proc=64,
        desc="Running tokenizer on every text in dataset",
    )
    print(tokenized_test_dataset)



    # Run evaluation
    model.eval()
    import torch.nn.functional as F
    from tqdm import tqdm

    tokenized_inputs = []
    for i in range(len(tokenized_test_dataset)):
        text = tokenized_test_dataset[i]['function_text']
        label = tokenized_test_dataset[i]['label']
        tokenized_input = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        tokenized_inputs.append((tokenized_input, text, label))

    # Sort by tokenized length
    tokenized_inputs.sort(key=lambda x: x[0]['input_ids'].shape[1])

    # Group the inputs by their tokenized length
    grouped_inputs = defaultdict(list)
    for tokenized_input, test_str, label in tokenized_inputs:
        length = tokenized_input['input_ids'].shape[1]
        grouped_inputs[length].append((tokenized_input, test_str, label))

    # Process each group in batches of 64
    batch_size = 16
    all_outputs = []
    all_strings = []
    all_labels = []
    all_probabilities = []

    for length, group in tqdm(grouped_inputs.items()):
        for i in range(0, len(group), batch_size):
            batch = group[i:i + batch_size]
            batch_inputs = [item[0] for item in batch]
            batch_strings = [item[1] for item in batch]
            batch_labels = [item[2] for item in batch]

            input_ids = torch.cat([item['input_ids'] for item in batch_inputs], dim=0).to("cuda:0")
            attention_mask = torch.cat([item['attention_mask'] for item in batch_inputs], dim=0).to("cuda:0")

            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits[:, -1, :2]
            probabilities = F.softmax(logits, dim=-1)
            predictions = (probabilities[:, 1] > 0.7).long()
            # predictions = torch.argmax(probabilities, dim=-1)
            all_probabilities.extend(probabilities[:, 1].cpu().tolist())
            all_outputs.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels)
            all_strings.extend(batch_strings)

    correct = 0
    total = 0
    ct_t = 0
    ct_f = 0
    correct_t = 0
    correct_f = 0
    pred_t = 0
    pred_f = 0
    import pickle
    with open('data.pkl', 'wb') as fp:
        pickle.dump((all_probabilities, all_labels), fp)

    for i in range(len(all_outputs)):
        pred = str(all_outputs[i])
        label = str(all_labels[i])
        # if i > len(all_outputs) - 25:
        #     print(f"{i}: text: {all_strings[i]}\n pred: {pred} label: {label}\n")

        if pred == label:
            correct += 1
        if int(pred) == 1:
            pred_t += 1
        else:
            pred_f += 1
        if int(label) == 1:
            ct_t += 1
            if int(pred) == 1:
                correct_t += 1
        else:
            ct_f += 1
            if int(pred) == 0:
                correct_f += 1
        total += 1

    print(f"Correct: {correct} Total: {total} Accuracy: {correct / total} {ct_t} {ct_f} {correct_t} {correct_f} {pred_t} {pred_f}")




