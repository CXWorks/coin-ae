# Vanilla LLM Few-Shot Baseline Prompt

Used in `code/llm_final.py` to evaluate closed-source frontier models (GPT-4o,
Claude-3.7) on the MUF detection task without fine-tuning. This is the baseline
reported in Table 2 of the paper (N=1,3,5 few-shot and Best@K evaluations).

## System Message

```
You are an experienced Rust developer. Help me validate whether the given Rust
function is safe or unsafe. Please think step by step with the given code and
context. Reply only `Yes` for unsafe or `No` for safe to proceed.
{few_shot_examples}
```

## User Message

```
The target code and relevant context is below, the target function is highlighted
by `>` at the beginning of the line.

```
{function_text}
```
Is the target function unsafe?
```

## Fields

- `{few_shot_examples}` — N labeled examples prepended to the system message
  (empty string for zero-shot). Each example is a (function_text, Yes/No) pair
  formatted the same way as the user message.
- `{function_text}` — the target function with surrounding context; target line
  prefixed with `>` (same format as Coin classifier).

## Notes

- The model is queried with `temperature=1` (default); for Best@K, queried K times
  and the majority vote or any-correct metric is used.
- Expected to require `OPENAI_API_KEY` environment variable (see `code/llm_final.py`).
- Results in the paper: GPT-4o 4.6% precision / 21.4% recall; Claude-3.7 4.2% / 7.1%
  (N=1 few-shot, Best@1), vs. Coin 63.7% / 80.4%.
