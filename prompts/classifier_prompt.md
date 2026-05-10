# Coin Classifier Prompt

Used during fine-tuning (`code/ft1.py`) and inference (`code/infer.py`) to format each
Rust function for the MUF classifier. The target function is prefixed with `>` in the
surrounding context. The model is trained to output `Yes` (unsafe / MUF) or `No` (safe).

## Template

```
Here is a Rust code and please check if the function starting with `>` is safe or unsafe:
{function_text}

Is this function unsafe? Answer with "Yes" or "No".

SOLUTION
The correct answer is: "{Yes|No}"
```

## Fields

- `{function_text}` — the target function with surrounding context (up to 8192 tokens).
  The target function line is prefixed with `>`.
- `{Yes|No}` — `Yes` if the function is a MUF (label=1), `No` otherwise (label=0).
  The answer token is omitted at inference time; the model's predicted token (`Yes`/`No`)
  is read from the logits of the first generated token.

## Notes

- The prompt is the same for both training (completion included) and inference (completion
  stripped after `The correct answer is: "`).
- PAC-based thresholding (§5.1.5) is applied post-inference on the log-probability of
  the `Yes` token to calibrate precision/recall.
