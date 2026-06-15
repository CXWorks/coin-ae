# needed as this function doesn't like it when the lm_head has its size changed
from unsloth import tokenizer_utils
def do_nothing(*args, **kwargs):
    pass
tokenizer_utils.fix_untrained_tokens = do_nothing

import torch
major_version, minor_version = torch.cuda.get_device_capability()
print(f"Major: {major_version}, Minor: {minor_version}")
import numpy as np
import os
from unsloth import FastLanguageModel
from typing import Any, List, Union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.environ["WANDB_DISABLED"] = "true"
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
max_seq_length = 8192
dtype = None

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, average_precision_score,
                             precision_recall_curve)

from torch.nn import Module
from collections import OrderedDict, defaultdict
from typing import Mapping, NamedTuple


def do_nothing(*args, **kwargs):
    pass


from unsloth import tokenizer_utils
tokenizer_utils.fix_untrained_tokens = do_nothing


def _find_mismatched_keys(
        model: torch.nn.Module, peft_model_state_dict: dict[str, torch.Tensor], ignore_mismatched_sizes: bool = True
) -> tuple[dict[str, torch.Tensor], list[tuple[str, tuple[int, ...], tuple[int, ...]]]]:
    return peft_model_state_dict, []


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

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = OrderedDict(state_dict)
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, local_state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        if assign:
            local_metadata["assign_to_params_buffers"] = assign
        module._load_from_state_dict(
            local_state_dict, prefix, local_metadata, True,
            missing_keys, unexpected_keys, error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + "."
                child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                load(child, child_state_dict, child_prefix)

        incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
        for hook in module._load_state_dict_post_hooks.values():
            out = hook(module, incompatible_keys)
            assert out is None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for name, param in self._parameters.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if param.shape != input_param.shape:
                    print(f"Shape mismatch for {key}, creating new tensor. Old shape: {param.shape}, New shape: {input_param.shape}")
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
                    print(f"Shape mismatch for buffer {key}, creating new tensor. Old shape: {buf.shape}, New shape: {input_buf.shape}")
                    new_buf = torch.empty_like(input_buf)
                    new_buf.copy_(input_buf)
                    setattr(self, name, new_buf)
                else:
                    buf.copy_(input_buf)
            elif strict:
                missing_keys.append(key)

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
        raise RuntimeError("Error(s) in loading state_dict for {}:\n\t{}".format(
            self.__class__.__name__, "\n\t".join(error_msgs)))

    return _IncompatibleKeys(missing_keys, unexpected_keys)


Module.load_state_dict = patched_load_state_dict


# Replicated exactly from data/coin/coin.py
def generate_prompt(text, st, ed, wst, wed):
    text = text.splitlines(keepends=True)
    for i in range(st - 1, ed):
        text[i] = '>\t' + text[i]
    if wst == st and wed == ed:
        return ''.join(text), st, ed
    else:
        return ''.join(text[wst - 1: wed]), st, ed


def generate_prompt_unsafe(text, st, ed, wst, wed):
    text = text.splitlines(keepends=True)
    text[st - 1] = text[st - 1].replace('unsafe ', '')
    for i in range(st - 1, ed):
        text[i] = '>\t' + text[i]
    if wst == st and wed == ed:
        return ''.join(text), st, ed
    else:
        return ''.join(text[wst - 1: wed]), st, ed


def load_model(checkpoint_path="outputs_sec/checkpoint-11000",
               base_model="unsloth/Llama-3.2-3B-bnb-4bit"):
    # Patch out the multi-GPU check: it calls nvidia-smi system-wide and sees all 8 GPUs
    # even when CUDA_VISIBLE_DEVICES restricts to one. Safe because we ARE running
    # single-GPU processes — just multiple separate processes on different GPUs.
    import unsloth.models.llama as _llama
    _llama.check_nvidia = lambda: np.array([0.0] * 8)
    model, tokenizer = FastLanguageModel.from_pretrained(checkpoint_path)
    return model, tokenizer


if __name__ == '__main__':
    import argparse, json, pickle, random
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='outputs_sec/checkpoint-11000')
    parser.add_argument('--data', default='/mnt/sdb/xiang/coin3/coin_test.pkl')
    parser.add_argument('--n', type=int, default=8000, help='number of samples to eval (0=all)')
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--output', default='/tmp/eval_repro_results.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shard_id', type=int, default=0, help='shard index (0-based)')
    parser.add_argument('--num_shards', type=int, default=1, help='total number of shards')
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint)
    FastLanguageModel.for_inference(model)

    PROMPT = """Here is a Rust code and please check if the function starting with `>` is safe or unsafe:
    {}

    Is this function unsafe? Answer with "Yes" or "No".

    SOLUTION
    The correct answer is: \""""

    # Load directly from coin_test.pkl — same logic as data/coin/coin.py _generate_examples_caller_callee
    print(f"Loading data from {args.data} ...")
    with open(args.data, 'rb') as fp:
        data = pickle.load(fp)

    examples = []  # list of (function_text_with_prompt, label)
    for k, vv in data.items():
        label = 0 if k == 'safe' else 1
        for f, text, ls, window in vv:
            for idx, (st, ed) in enumerate(ls):
                wst, wed = window[idx][0], window[idx][1]
                if label == 0:
                    prompt_text, _, _ = generate_prompt(text, st, ed, wst, wed)
                else:
                    prompt_text, _, _ = generate_prompt_unsafe(text, st, ed, wst, wed)
                full_prompt = PROMPT.format(prompt_text)
                examples.append((full_prompt, label))

    n_total = len(examples)
    n_safe = sum(1 for _, l in examples if l == 0)
    n_unsafe = sum(1 for _, l in examples if l == 1)
    print(f"Total examples: {n_total} (safe={n_safe}, unsafe={n_unsafe})")

    if args.n > 0 and args.n < n_total:
        random.seed(args.seed)
        examples = random.sample(examples, args.n)
        n_safe = sum(1 for _, l in examples if l == 0)
        n_unsafe = sum(1 for _, l in examples if l == 1)
        print(f"Sampled {args.n} examples (safe={n_safe}, unsafe={n_unsafe})")

    # Shard the examples for distributed evaluation
    if args.num_shards > 1:
        # Shuffle with fixed seed first so each shard gets proportional safe/unsafe mix
        random.seed(args.seed)
        random.shuffle(examples)
        shard_size = (len(examples) + args.num_shards - 1) // args.num_shards
        start = args.shard_id * shard_size
        end = min(start + shard_size, len(examples))
        examples = examples[start:end]
        n_safe = sum(1 for _, l in examples if l == 0)
        n_unsafe = sum(1 for _, l in examples if l == 1)
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(examples)} examples (safe={n_safe}, unsafe={n_unsafe})")

    # Sort by tokenized length (batch-by-length for efficiency)
    import torch.nn.functional as F
    from tqdm import tqdm

    tokenized_inputs = []
    for prompt_text, label in examples:
        tokenized_input = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False,
                                    max_length=max_seq_length, truncation=True)
        tokenized_inputs.append((tokenized_input, prompt_text, label))

    tokenized_inputs.sort(key=lambda x: x[0]['input_ids'].shape[1])

    grouped_inputs = defaultdict(list)
    for tokenized_input, text_str, label in tokenized_inputs:
        length = tokenized_input['input_ids'].shape[1]
        grouped_inputs[length].append((tokenized_input, text_str, label))

    batch_size = 16
    all_outputs = []
    all_strings = []
    all_labels = []
    all_probabilities = []

    model.eval()
    for length, group in tqdm(grouped_inputs.items()):
        for i in range(0, len(group), batch_size):
            batch = group[i:i + batch_size]
            batch_inputs = [item[0] for item in batch]
            batch_strings = [item[1] for item in batch]
            batch_labels = [item[2] for item in batch]

            input_ids = torch.cat([item['input_ids'] for item in batch_inputs], dim=0).to("cuda:0")
            attention_mask = torch.cat([item['attention_mask'] for item in batch_inputs], dim=0).to("cuda:0")

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits[:, -1, :2]
            probabilities = F.softmax(logits, dim=-1)
            predictions = (probabilities[:, 1] > args.threshold).long()
            all_probabilities.extend(probabilities.cpu().to(torch.float32).numpy())
            all_outputs.extend(predictions.cpu().to(torch.float32).numpy())
            all_labels.extend(batch_labels)
            all_strings.extend(batch_strings)

    shard_suffix = f'_shard{args.shard_id}' if args.num_shards > 1 else ''
    with open(f'data{shard_suffix}.pkl', 'wb') as fp:
        pickle.dump((all_probabilities, all_labels), fp)

    # Metrics
    all_probs_unsafe = [float(p[1]) for p in all_probabilities]
    all_labels_arr = np.array(all_labels)
    all_preds_arr = np.array(all_outputs, dtype=int)

    correct = ct_t = ct_f = correct_t = correct_f = pred_t = pred_f = 0
    total = len(all_outputs)
    for i in range(total):
        pred = int(all_outputs[i])
        label = int(all_labels[i])
        if pred == label:
            correct += 1
        pred_t += pred
        pred_f += 1 - pred
        if label == 1:
            ct_t += 1
            correct_t += int(pred == 1)
        else:
            ct_f += 1
            correct_f += int(pred == 0)

    accuracy = correct / total
    print(f"Correct: {correct} Total: {total} Accuracy: {accuracy:.4f} "
          f"ct_t={ct_t} ct_f={ct_f} correct_t={correct_t} correct_f={correct_f} "
          f"pred_t={pred_t} pred_f={pred_f}")

    auprc = average_precision_score(all_labels_arr, all_probs_unsafe)
    prec_curve, rec_curve, thresholds = precision_recall_curve(all_labels_arr, all_probs_unsafe)
    print(f"AUPRC: {auprc:.4f}")

    cm = confusion_matrix(all_labels_arr, all_preds_arr)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels_arr, all_preds_arr, average=None, labels=[0, 1])
    print(f"Precision safe={prec[0]:.4f} unsafe={prec[1]:.4f}")
    print(f"Recall    safe={rec[0]:.4f} unsafe={rec[1]:.4f}")
    print(f"F1        safe={f1[0]:.4f} unsafe={f1[1]:.4f}")
    print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

    # AUPRC plot
    out_png = args.output.replace('.json', '_auprc.png')
    plt.figure(figsize=(8, 6))
    plt.plot(rec_curve, prec_curve, label=f'AUPRC={auprc:.4f}')
    if len(thresholds) > 0:
        op_idx = int(np.argmin(np.abs(thresholds - args.threshold)))
        plt.scatter([rec_curve[op_idx]], [prec_curve[op_idx]], color='red', zorder=5,
                    label=f'threshold={args.threshold}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"AUPRC plot saved to {out_png}")

    results = {
        'checkpoint': args.checkpoint,
        'data': args.data,
        'n_samples': total,
        'n_safe': int(ct_f),
        'n_unsafe': int(ct_t),
        'threshold': args.threshold,
        'seed': args.seed,
        'accuracy': float(accuracy),
        'auprc': float(auprc),
        'precision_safe': float(prec[0]),
        'recall_safe': float(rec[0]),
        'f1_safe': float(f1[0]),
        'precision_unsafe': float(prec[1]),
        'recall_unsafe': float(rec[1]),
        'f1_unsafe': float(f1[1]),
        'macro_f1': float(np.mean(f1)),
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
    }
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")
