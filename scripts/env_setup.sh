#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Environment Setup for Go-Browse LoRA Fine-tuning
# Target: Vast.ai instance with 4× A100 40GB
# Model: Qwen/Qwen3.5-9B
# ============================================================

echo "=== Go-Browse Training Environment Setup ==="

# --- Verify CUDA ---
echo "[1/6] Checking CUDA..."
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')" || true

# --- System dependencies ---
echo "[2/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq git curl wget build-essential 2>&1 | tail -1

# --- PyTorch with CUDA 12.4 ---
echo "[3/6] Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# --- Flash Attention 2 ---
# Pre-built wheel is fastest; falls back to source build
echo "[4/6] Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation || {
    echo "WARN: flash-attn wheel failed, building from source (5-10 min)..."
    MAX_JOBS=4 pip install flash-attn --no-build-isolation
}

# --- Core training stack ---
echo "[5/6] Installing training dependencies..."
pip install \
    accelerate>=1.0.0 \
    deepspeed>=0.15.0 \
    bitsandbytes>=0.44.0 \
    transformers>=4.46.0 \
    trl>=0.12.0 \
    peft>=0.13.0 \
    datasets>=3.0.0 \
    wandb>=0.18.0 \
    tensorboard>=2.18.0 \
    sentencepiece>=0.2.0 \
    protobuf>=5.0.0 \
    huggingface_hub>=0.26.0 \
    fsspec>=2024.0.0

# --- Project deps ---
echo "[6/6] Installing project dependencies..."
pip install agentlab browsergym evaluate kaleido scikit-learn liger-kernel omegaconf pillow plotly

# Install the project itself (editable)
pip install -e .

# --- Verify ---
echo ""
echo "=== Verification ==="
python -c "
import torch; print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPUs: {torch.cuda.device_count()}')
import flash_attn; print(f'Flash Attention {flash_attn.__version__}')
import deepspeed; print(f'DeepSpeed {deepspeed.__version__}')
import transformers; print(f'Transformers {transformers.__version__}')
import trl; print(f'TRL {trl.__version__}')
import peft; print(f'PEFT {peft.__version__}')
import bitsandbytes; print(f'bitsandbytes {bitsandbytes.__version__}')
"

echo ""
echo "=== Environment setup complete ==="
echo "Next: python preprocess.py"
echo "Then: bash scripts/launch_train.sh"
