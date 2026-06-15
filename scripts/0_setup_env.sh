#!/bin/bash
# Set up the conda env with the pinned package versions required for AE.
# CRITICAL: unsloth must be 2025.2.15 — newer versions silently truncate
# sequences and degrade classifier AUPRC from ~0.76 to ~0.32.

set -e
ENV_PATH="${ENV_PATH:-./env}"

echo "Creating conda env at $ENV_PATH ..."
conda create -p "$ENV_PATH" python=3.11 -y

echo "Installing PyTorch (CUDA 12.1)..."
"$ENV_PATH/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu121

echo "Installing pinned unsloth + transformers + peft + trl..."
"$ENV_PATH/bin/pip" install "unsloth==2025.2.15"
"$ENV_PATH/bin/pip" install --no-deps \
    "transformers==4.49.0" "peft==0.14.0" "trl==0.15.2"
"$ENV_PATH/bin/pip" install --no-deps \
    "tokenizers==0.21.0" scikit-learn matplotlib numpy tqdm rich

echo "Installing API baseline drivers (E4)..."
"$ENV_PATH/bin/pip" install "openai>=1.30" "anthropic>=0.30" "bitsandbytes>=0.43"

echo
echo "Verifying installation:"
"$ENV_PATH/bin/python" -c "import unsloth; print('unsloth:', unsloth.__version__)"
"$ENV_PATH/bin/python" -c "import transformers; print('transformers:', transformers.__version__)"

echo
echo "Done. To run scripts:"
echo "  export ENV_PATH=$ENV_PATH"
echo "  bash scripts/1_smoke_test.sh /path/to/coin_test.pkl"
