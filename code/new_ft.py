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
    DataCollatorWithPadding, DataCollatorForWholeWordMask, RobertaForSequenceClassification, Trainer
import os
import sys
from torch.nn import CrossEntropyLoss
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from args import ModelArguments, DataTrainingArguments

checkpoint = "microsoft/codebert-base"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

os.environ["WANDB_DISABLED"] = "false"
logger = logging.getLogger(__name__)


def init_model():
    model = RobertaForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
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
        ans = tokenizer(row["function_text"], padding='max_length', truncation=True)
        ans['label'] = 0 if row["label"] == 'safe' else 1
        return ans

    with training_args.main_process_first(desc="dataset map tokenization"):
        num_proc = data_args.preprocessing_num_workers
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            num_proc=num_proc,
            batched=False,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
    # tokenized_datasets["test"] = tokenized_datasets["test"].shuffle().select(
    #     range(10))
    # tokenized_datasets["validation"] = tokenized_datasets["validation"].shuffle().select(
    #     range(10000))
    for i in range(1):
        print('=' * 20)
        print(tokenized_datasets['train'][i])

    return raw_datasets, tokenized_datasets


def init_tokenizer(model_args, data_args, training_args):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer


from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import scipy


# Define metrics computation function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', pos_label=1)
    sprecision, srecall, sf1, _ = precision_recall_fscore_support(labels, preds, average='binary', pos_label=0)
    accuracy = accuracy_score(labels, preds)

    return {
        'accuracy': accuracy,
        'unsafe_precision': precision,
        'unsafe_recall': recall,
        'unsafe_f1': f1,
        'safe_precision': sprecision,
        'safe_recall': srecall,
        'safe_f1': sf1,
    }


# Compute class weights
def compute_class_weights(tokenized_datasets, label_column='label'):
    labels = tokenized_datasets['train'][label_column]
    class_weights = compute_class_weight('balanced', classes=[0, 1], y=labels)
    return class_weights


# Define custom loss function
def weighted_loss_function(class_weights):
    class_weights_tensor = torch.FloatTensor(class_weights).to('cuda')  # Adjust for your device

    def loss_fn(labels, logits):
        loss_fct = CrossEntropyLoss(weight=class_weights_tensor)
        return loss_fct(logits, labels)

    return loss_fn


# Define custom Trainer
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = weighted_loss_function(class_weights) if class_weights is not None else CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs['labels'].to('cuda')  # Ensure labels are on the same device
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


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
    logger.info("***** Running training *****")
    logger.info("num train epochs: %d", training_args.num_train_epochs)
    class_weights = compute_class_weights(tokenized_datasets)
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=training_args.output_dir,
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
        fp16=True
    )
    # Create a Seq2SeqTrainer

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
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
    os.environ["WANDB_PROJECT"] = "rust_ft"

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

