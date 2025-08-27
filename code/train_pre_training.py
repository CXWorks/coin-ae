import torch
from datasets import load_dataset
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, \
    AutoModelForSeq2SeqLM, DataCollatorForLanguageModeling
import os
import sys

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

checkpoint = "Salesforce/codet5p-770m"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

os.environ["WANDB_DISABLED"] = "true"
dataset = load_dataset("EdinburghNLP/xsum")

config = T5Config.from_json_file('config-770m.json')
print(config, tokenizer)
# Initialize a new model from the configuration
model = T5ForConditionalGeneration(config)

model.to(device)
# Tokenize the input dataset
def preprocess_function(examples):
    inputs = examples['document']
    targets = examples['summary']
    model_inputs = tokenizer(inputs, truncation=True, padding='max_length')
    labels = tokenizer(targets, truncation=True, padding='max_length')
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



# Define the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=False
)
with training_args.main_process_first(desc="dataset map tokenization"):
    tokenized_datasets = dataset.map(preprocess_function, num_proc=96)

print(tokenized_datasets)
print(tokenized_datasets['train'][0])
# Create a Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train the model
trainer.train()

# Save the model after training
trainer.save_model("./trained-scratch-codet5p")
