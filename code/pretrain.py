import logging
from dataclasses import dataclass
from typing import Any, Tuple

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, \
    AutoModelForSeq2SeqLM, DataCollatorForLanguageModeling, HfArgumentParser, TrainingArguments, set_seed, \
    DataCollatorWithPadding, DataCollatorForWholeWordMask
import os
import sys

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from args import ModelArguments, DataTrainingArguments

checkpoint = "Salesforce/codet5p-770m"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

os.environ["WANDB_DISABLED"] = "false"
logger = logging.getLogger(__name__)


def init_model():
    config = T5Config.from_json_file('config-220m.json')
    # Initialize a new model from the configuration
    model = T5ForConditionalGeneration(config)
    model.to(device)
    return model


def init_datasets(model_args, data_args, training_args, tokenizer):
    # load raw datasets
    assert (data_args.dataset_name)
    raw_datasets = load_dataset(
        data_args.dataset_name,
        "caller",
        cache_dir=model_args.cache_dir,
    )

    # We tokenize each function in asm
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
    for i in range(1):
        print('=' * 20)
        print(tokenized_datasets['train'][i])
    # tokenized_datasets["test"] = tokenized_datasets["test"].shuffle().select(
    #     range(10))
    # tokenized_datasets["validation"] = tokenized_datasets["validation"].shuffle().select(
    #     range(10))
    return raw_datasets, tokenized_datasets


def init_tokenizer(model_args, data_args, training_args):
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    print(model_args.tokenizer_name)
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


def mlm_accuracy(eval_preds):
    predictions, labels = eval_preds
    sp = predictions.shape
    # predictions = np.argmax(predictions, axis=-1)
    x, y = predictions.shape
    if y < 100:
        predictions = np.pad(predictions, pad_width=((0, 0), (0, 100 - y)), constant_values=-100)
    elif y > 100:
        predictions = predictions[:, :100]
    valid_mask = labels != -100  # Only consider masked tokens, ignoring padded tokens labeled as -100

    acc = (predictions[valid_mask] == labels[valid_mask]).astype(np.float32)
    logger.warning('MLM Acc: {}, shape: {}'.format(len(acc), sp))
    return {'accuracy': acc.mean()}


def train_and_eval(tokenized_datasets, model, tokenizer, model_args, data_args, training_args):
    print("Training set size:", len(tokenized_datasets['train']))
    print("Validation set size:", len(tokenized_datasets['validation']))
    print("Test set alternatives:", len(tokenized_datasets['test']))
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # init a collator for classification
    data_collator_cls = CustomDataCollector(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,

    )
    logger.info("***** Running training *****")
    logger.info("num train epochs: %d", training_args.num_train_epochs)
    # Define the training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=training_args.output_dir,
        generation_max_length=100,
        evaluation_strategy=training_args.evaluation_strategy,
        eval_steps=training_args.eval_steps,
        save_strategy=training_args.save_strategy,
        report_to=training_args.report_to,
        save_steps=training_args.save_steps,
        learning_rate=training_args.learning_rate,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        weight_decay=training_args.weight_decay,
        save_total_limit=training_args.save_total_limit,
        num_train_epochs=training_args.num_train_epochs,
        predict_with_generate=True,
        fp16=True
    )

    print(tokenized_datasets, len(tokenized_datasets['train']), training_args.generation_max_length)
    # Create a Seq2SeqTrainer

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator_cls,
        tokenizer=tokenizer,
        compute_metrics=mlm_accuracy
    )

    if True:

        logger.info("*** Train an unsafe classifier ***")
        # train
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate an unsafe classifier on a test dataset***")

        metrics = trainer.evaluate(eval_dataset=tokenized_datasets['test'], metric_key_prefix="test_")
        metrics["test_samples"] = len(tokenized_datasets['test'])

        trainer.log_metrics("test_accuracy", metrics)
        trainer.save_metrics("test_accuracy", metrics)

    return model


# ==================================================
# main
# ==================================================
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
    )
    log_level = 'WARNING'
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
    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"] = "rust_mlm_pretrain"

    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"] = "false"

    # turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false"

    # save stdout and stderr
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Init a tokenizer
    tokenizer = init_tokenizer(model_args, data_args, training_args)

    # init datasets
    raw_datasets, tokenized_datasets = init_datasets(model_args, data_args, training_args, tokenizer)

    # # Init a model
    model = init_model()

    train_and_eval(tokenized_datasets, model, tokenizer, model_args, data_args, training_args)


if __name__ == "__main__":
    main()

