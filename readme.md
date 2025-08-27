# AE for COIN: Detecting Logically unsafe Rust Function

The ae of coin includes four parts:

1. The `code` folder includes the code to train/fine-tune/evaluate the model and output result
2. The `custom_rustc_patch` contains the patch diff for our customized Rust toolchain: 1.83.0-dev
3. The `data` folder includes the logically unsafe Rust dataset we used in training process: train/validation/test
4. The `model` folder includes the model weights after fine-tuning. Together with the original model weights to work. Original model can be accessed through HF. 