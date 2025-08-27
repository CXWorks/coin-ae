import openai
from openai import OpenAI
import os
import sys
from tqdm import tqdm
import pickle


def run_llm(client: OpenAI, model, examples, target_text, best):
    preds = []
    for _ in range(best):
        msgs = [
            {
                "role": "system",
                "content": (
                    "You are an experienced Rust developer. Help me validate whether the given Rust function is safe or unsafe. "
                    # "Safe Rust will never trigger undefined behavior. Even without unsafe operations, the function can be unsafe. "
                    "Please think step by step with the given code and context. Reply only `Yes` for unsafe or `No` for safe to proceed.\n"
                    f"{examples}"
                )
            },
            {
                "role": "user",
                "content": (
                    "The target code and relevant context is below, the target function is highlighted by `>` at the beginning of the line.\n\n"
                    "```\n"
                    f"{target_text}\n"
                    "```\n"
                    "Is the target function unsafe?"
                )
            }
        ]
        completion = client.chat.completions.create(
            model=model,
            messages=msgs
        )
        ans = completion.choices[0].message.content.lower()
        print(ans)
        if 'yes' in ans:
            preds.append(1)
        elif 'no' in ans:
            preds.append(0)
        else:
            preds.append(-1)
    return preds


def build_client():
    gpt4o = openai.OpenAI(
        api_key='sk-w7weiFTL6mzJrr3P4S3wEA',
        base_url="https://litellm-proxy-153298433405.us-east1.run.app/"
    )
    return gpt4o


if __name__ == '__main__':
    models = ['gpt-4o', 'claude-3-7-sonnet-20250219']
    client = build_client()

    data = {}
    for model_name in models:
        print(f"\n=== Running LLM classification with model: {model_name} ===\n")

        tp = 0  # true positives (predicted unsafe, actually unsafe)
        fp = 0  # false positives (predicted unsafe, actually safe)
        fn = 0  # false negatives (predicted safe, actually unsafe)
        total = 0

        # Store all (true_label, predicted_label) pairs if needed
        cur = []

        for idx in range(20):
            target_code = open(f'final/{idx+1}.txt', 'r').read()
            true_label = 1

            # Run LLM best times
            preds = run_llm(client, model_name, '', target_code, best=1)

            predicted_label = preds[0]
            print(model_name, idx+1, predicted_label)

            cur.append((true_label, preds))

            # Update TP, FP, FN
            if predicted_label == 1:
                if true_label == 1:
                    tp += 1
                else:
                    fp += 1
            else:  # predicted_label == 0
                if true_label == 1:
                    fn += 1

            total += 1

        # Compute precision and recall for unsafe label (label=1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        print(f"Model {model_name} results on {total} examples:")
        print(f"  True Positives (unsafe & predicted unsafe): {tp}")
        print(f"  False Positives (safe & predicted unsafe): {fp}")
        print(f"  False Negatives (unsafe & predicted safe): {fn}")
        print(f"  Precision (unsafe): {precision:.4f}")
        print(f"  Recall    (unsafe): {recall:.4f}\n")

        data[model_name] = cur


    print('done')
