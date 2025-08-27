#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Any, Tuple
import warnings
import numpy as np
import scipy
import pickle
import types

import torch as tc

import datasets
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DefaultDataCollator,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed, DataCollatorForWholeWordMask,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version  # , send_example_telemetry
from transformers.utils.versions import require_version

import utils

from args import *

# from models import (
#     CallerUnsafeClassification,
#     CallerCalleeAveUnsafeClassification
# )

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.21.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


def one_hot_unsafe(examples, n_classes):
    unsafe_label = examples['unsafe_label']
    multilabel_list = []
    assert (len(unsafe_label))
    for l_multi in unsafe_label:
        label = [0.0] * n_classes
        for l in l_multi:
            label[l] = 1.0
        multilabel_list.append(label)
    return {'label': multilabel_list}


def init_tokenizer(model_args, data_args, training_args):
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    return tokenizer


@dataclass
class CustomDataCollector(DataCollatorForWholeWordMask):

    def torch_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        import torch

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% -> 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.9)).bool() & masked_indices
        # inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        for inp, idx in zip(inputs, indices_replaced):
            num_true = idx.sum()
            candidates = torch.tensor(
                self.tokenizer.convert_tokens_to_ids([f'<extra_id_{x}>' for x in range(num_true)]), dtype=torch.int64)
            inp[idx] = candidates
        filtered_results = [row[mask].tolist() for row, mask in zip(labels, masked_indices)]

        # Pad the results to the desired length
        padded_results = [row + [-100] * (100 - len(row)) for row in filtered_results] 
        # filtered_results = [list(compressed_row) for row, mask in zip(labels, masked_indices) for compressed_row in [row[mask]]]

        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        # inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, torch.tensor(padded_results)


def init_datasets(model_args, data_args, training_args):
    # load raw datasets
    assert (data_args.dataset_name)
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained('/mnt/md0/xiang/RustUnsafeDetector/rust_tokenizer')
    def tokenize_function(row):
        return tokenizer(row['function_text'], truncation=True, return_special_tokens_mask=True, max_length=512)

    with training_args.main_process_first(desc="dataset map tokenization"):
        num_proc = data_args.preprocessing_num_workers
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            num_proc=num_proc,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
    for i in range(10):
        print('=' * 20)
        print(tokenized_datasets['train'][i])
    data_collator = CustomDataCollector(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    mlm_batch = data_collator(tokenized_datasets['test'])

    # Print the batch to see the result
    input_ids = mlm_batch['input_ids']
    labels = mlm_batch['labels']

    for idx, (input_id, label) in enumerate(zip(input_ids, labels)):
        if idx > 10:
            break
        print(f"Example {idx + 1}:")
        print("Input IDs:", input_id.tolist())
        print("Labels:", label.tolist())
        decoded_inputs = tokenizer.decode(input_id, skip_special_tokens=False)
        decoded_labels = [
            tokenizer.decode(label[i].unsqueeze(0), skip_special_tokens=False) if label[i] != -100 else "not masked" for
            i in range(len(label))]
        print("Decoded Inputs:", decoded_inputs)
        print("Decoded Labels:", ' '.join(decoded_labels))
    return raw_datasets, tokenized_datasets



    # tokenize datasets
    # column_names = raw_datasets["train"].column_names
    # text_column_name = "text" if "text" in column_names else column_names[0]
    #
    # if data_args.max_seq_length is None:
    #     max_seq_length = tokenizer.model_max_length
    #     if max_seq_length > 1024:
    #         logger.warning(
    #             f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
    #             "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
    #         )
    #         max_seq_length = 1024
    # else:
    #     if data_args.max_seq_length > tokenizer.model_max_length:
    #         logger.warning(
    #             f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
    #             f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    #         )
    #     max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    #
    # # We tokenize each function in asm
    # def tokenize_function(examples):
    #     t_list = [tokenizer(e, return_special_tokens_mask=True, truncation=True) for e in examples[text_column_name]]
    #     out = {k: [v] for k, v in t_list[0].items()}
    #     for t_i in t_list[1:]:
    #         for k, v in t_i.items():
    #             out[k].append(v)
    #     return out
    #
    # with training_args.main_process_first(desc="dataset map tokenization"):
    #     num_proc = data_args.preprocessing_num_workers
    #     tokenized_datasets = raw_datasets.map(
    #         tokenize_function,
    #         batched=True,
    #         num_proc=num_proc,
    #         remove_columns=[c for c in column_names if 'label' not in c],
    #         load_from_cache_file=not data_args.overwrite_cache,
    #         desc="Running tokenizer on every text in dataset",
    #     )
    #
    # with training_args.main_process_first(desc="dataset map for label transformations"):
    #     num_unsafe_classes = model_args.num_unsafe_classes
    #     tokenized_datasets = tokenized_datasets.map(
    #         lambda x: one_hot_unsafe(x, num_unsafe_classes),
    #         batched=True,
    #         num_proc=data_args.preprocessing_num_workers,
    #         load_from_cache_file=not data_args.overwrite_cache,
    #         desc="convertion from a list to a one-hot label vector"
    #     )
    #
    # # truncate samples with sampling
    # # if data_args.max_train_samples is not None:
    # #     max_train_samples = min(len(tokenized_datasets["train"]), data_args.max_train_samples)
    # #     tokenized_datasets["train"] = tokenized_datasets["train"].shuffle(seed=training_args.seed).select(
    # #         range(max_train_samples))
    # # if data_args.max_validation_samples is not None:
    # #     max_validation_samples = min(len(tokenized_datasets["validation"]), data_args.max_validation_samples)
    # #     tokenized_datasets["validation"] = tokenized_datasets["validation"].shuffle(seed=training_args.seed).select(
    # #         range(max_validation_samples))
    # # if data_args.max_test_samples is not None:
    # #     max_test_samples = min(len(tokenized_datasets["test"]), data_args.max_test_samples)
    # #     tokenized_datasets["test"] = tokenized_datasets["test"].shuffle(seed=training_args.seed).select(
    # #         range(max_test_samples))
    # print(tokenized_datasets['train'][0])
    # count = 0
    # total = len(tokenized_datasets['test'])
    # for data in tokenized_datasets['test']:
    #     if data['label'][0] < 0.5:
    #         count += 1
    # print('=' * 20, count, total)
    # print(
    #     f'[datasets] #train = {len(tokenized_datasets["train"])}, #val = {len(tokenized_datasets["validation"])}, #test = {len(tokenized_datasets["test"])}')
    #
    # print(tokenized_datasets['train'][0])



# ==================================================
# main
# ==================================================
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # save stdout and stderr
    os.makedirs(training_args.output_dir, exist_ok=True)
    sys.stdout = utils.Logger(os.path.join(training_args.output_dir, 'out'))
    sys.stderr = utils.Logger(os.path.join(training_args.output_dir, 'out_err'))

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Init a tokenizer
    # tokenizer = init_tokenizer(model_args, data_args, training_args)

    # # Init a model
    # model = init_model(tokenizer, model_args, data_args, training_args)

    # Init datasets
    raw_datasets, _ = init_datasets(model_args, data_args, training_args)


if __name__ == "__main__":
    main()

