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
    DataCollatorWithPadding, DataCollatorForWholeWordMask, AutoModelForSequenceClassification, Trainer
import os
import sys

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from args import ModelArguments, DataTrainingArguments

checkpoint = "Salesforce/codet5-large"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

os.environ["WANDB_DISABLED"] = "true"
logger = logging.getLogger(__name__)


def init_model():
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
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
        ans = tokenizer("Classify the function is safe or unsafe:\n" + row["function_text"], padding='max_length', truncation=True)
        ans['label'] = int(row["label"])
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
    #     range(10))
    for i in range(1):
        print('=' * 20)
        print(tokenized_datasets['train'][i])

    return raw_datasets, tokenized_datasets


def init_tokenizer(model_args, data_args, training_args):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer


from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import scipy

def compute_metrics(p):
    """
    Computes metrics for binary classification.

    Args:
    p (EvalPrediction): An instance of EvalPrediction that contains the predictions and the labels.

    Returns:
    dict: A dictionary with the computed metrics (accuracy, precision, recall, f1).
    """
    predictions, labels = p
    print(predictions[0], labels, predictions[1])
    preds1 = np.argmax(predictions[0], axis=1)
    # preds2 = (np.squeeze(predictions[0][:, 1]) > 0).astype(int)
    # probs = scipy.special.softmax(predictions[0], axis=1)
    # Predict the class with the highest probability
    # preds3 = np.argmax(probs, axis=1)


    accuracy1 = accuracy_score(labels, preds1)
    precision1, recall1, f11, _ = precision_recall_fscore_support(labels, preds1, average='binary')
    # accuracy2 = accuracy_score(labels, preds2)
    # precision2, recall2, f12, _ = precision_recall_fscore_support(labels, preds2, average='binary')
    # accuracy3 = accuracy_score(labels, preds3)
    # precision3, recall3, f13, _ = precision_recall_fscore_support(labels, preds3, average='binary')

    ans = {
        'accuracy': accuracy1,
        'precision': precision1,
        'recall': recall1,
        'f1': f11
    }
    # ans2 = {
    #     'accuracy': accuracy2,
    #     'precision': precision2,
    #     'recall': recall2,
    #     'f1': f12
    # }
    # ans3 = {
    #     'accuracy': accuracy3,
    #     'precision': precision3,
    #     'recall': recall3,
    #     'f1': f13
    # }
    logger.warning(f'{ans}')
    return ans


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

    trainer = Trainer(
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

