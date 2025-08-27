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
from typing import Optional
import warnings
import numpy as np
import scipy
import pickle
import types

import torch as tc

import datasets
from datasets import load_dataset

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
    set_seed,
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


def init_datasets(model_args, data_args):
    # load raw datasets
    assert (data_args.dataset_name)
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
    )



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

    return raw_datasets, tokenized_datasets


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
    raw_datasets, tokenized_datasets = init_datasets(model_args, data_args)


if __name__ == "__main__":
    main()
