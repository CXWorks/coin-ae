# Coin PoC Generator Prompt

Used during fine-tuning (`code/train_poc_generator.py`) and inference to format each
MUF for the PoC generation model. The model is trained to produce a complete PoC
(Cargo.toml + main.rs) that demonstrates undefined behavior via an all-safe caller.

Completion loss is applied only to the `### Response:` section and onwards
(via `DataCollatorForCompletionOnlyLM` with `response_template="### Response:\n"`).

## Template

```
### Instruction:
The following Rust function is a Modular Unsafe Function (MUF) of category "{category}".
Write a minimal PoC (Cargo.toml + main.rs) that demonstrates undefined behavior
by violating its invariant.

Function:
```rust
{function_text}
```

### Response:
### Explanation:
{explanation}

### Cargo.toml:
```toml
{poc_cargo_toml}
```

### src/main.rs:
```rust
{poc_main_rs}
```

### Verification command:
```
cargo +nightly miri run
```
```

## Fields

- `{category}` — one of the seven MUF root-cause categories:
  `logical requirement`, `ffi`, `logical memory controls`, `sharing status`,
  `steal reference`, `embedding memory mapping`, `hardware feature`
- `{function_text}` — the unsafe safe function body as a Rust code block
- `{explanation}` — natural-language description of the invariant violation
- `{poc_cargo_toml}` — minimal `Cargo.toml` with the target crate as a dependency
- `{poc_main_rs}` — the PoC `main.rs`; must contain no `unsafe` blocks in caller code
  where possible (soundness bugs should be triggerable from safe Rust)

## Notes

- At inference time, strip everything from `### Response:` onward; the model generates the full response.
- Verify generated PoCs with `cargo +nightly miri run` before accepting as confirmed bugs.
