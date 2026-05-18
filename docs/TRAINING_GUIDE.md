# Go-Browse LoRA Fine-tuning Guide

How to fine-tune Qwen3.5-9B on `apurvaga/go-browse-wa` using 4×A100 40GB on [Vast.ai](https://vast.ai).

## Overview

We fine-tune a LoRA adapter on top of Qwen3.5-9B to produce web agent trajectories. The dataset contains structured exploration traces from the Go-Browse algorithm: each sample pairs a prompt (system instructions + page accessibility tree + action space) with a completion (the agent's JSON action).

| Component | Choice |
|-----------|--------|
| Base model | `Qwen/Qwen3.5-9B` |
| Dataset | `apurvaga/go-browse-wa` |
| Method | LoRA (r=16, alpha=32) |
| Precision | bf16 |
| Attention | FlashAttention 2 |
| Distributed | DeepSpeed ZeRO-2, 4 GPUs |
| Max seq length | 8192 tokens |
| Packing | Enabled |
| Epochs | 2 |
| Effective batch | 16 (1 per GPU × 4 GPUs × 4 accum) |

Expected training time: **1-3 hours** depending on dataset size.

---

## Step 1: Rent a Vast.ai instance

### Instance requirements

| Spec | Minimum | Recommended |
|------|---------|-------------|
| GPUs | 4× A100 40GB | 4× A100 80GB |
| VRAM per GPU | 30 GB free | 40+ GB free |
| CPU cores | 16 | 32 |
| RAM | 64 GB | 128 GB |
| Disk | 100 GB | 200 GB |
| CUDA | 12.4 | 12.4 |
| Image | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` | same |

### How to rent

1. Go to [vast.ai/console/create](https://vast.ai/console/create/)
2. Filter: `A100` `4× GPU` `verified` `≥100GB disk`
3. Sort by `$/hr` (typical: $3-6/hr for 4×A100)
4. Select the `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` Docker image
5. Check "Run a command" and leave blank (we'll SSH in)
6. Click **Rent**

Once the instance is running, copy the SSH command from the Vast.ai console:

```bash
ssh -p <PORT> root@<HOST> -L 8080:localhost:8080
```

The `-L 8080:localhost:8080` port forward lets you access Jupyter/TensorBoard locally if needed.

---

## Step 2: Clone and set up

SSH into the instance and run:

```bash
git clone https://github.com/<your-org>/Go-Browse.git
cd Go-Browse
bash scripts/env_setup.sh
```

This installs everything in order:
1. PyTorch with CUDA 12.4
2. FlashAttention 2
3. DeepSpeed, Transformers, TRL, PEFT, bitsandbytes
4. Project dependencies + editable install

**Takes ~5-10 minutes.** FlashAttention may compile from source if no pre-built wheel matches — this is normal and takes ~5 minutes extra.

Verify the setup:

```bash
python -c "
import torch; print(f'GPUs: {torch.cuda.device_count()}')
import flash_attn; print(f'FlashAttn: {flash_attn.__version__}')
import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')
"
```

Expected output: `GPUs: 4` with version numbers for each library.

---

## Step 3: Preprocess the dataset

### From HuggingFace (full dataset)

```bash
python preprocess.py --full
```

This:
1. Downloads `apurvaga/go-browse-wa` from HuggingFace
2. Filters to positive-reward trajectories (`traj_reward > 0`)
3. Extracts `step_data` (prompt + completion message lists)
4. Removes malformed or empty samples
5. Flattens into `messages` format
6. Creates 95/5 train/validation split
7. Saves to `processed_data/train/` and `processed_data/val/`

### From local copy (if you pre-downloaded)

```bash
python preprocess.py --local data_local/go-browse-wa
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--full` | false | Process entire dataset (without this: first 50 rows for testing) |
| `--val-size` | 0.05 | Validation split fraction |
| `--num-proc` | 8 | Parallel workers for map/filter |
| `--seed` | 42 | Random seed for train/val split |
| `--output-dir` | `processed_data` | Where to save processed splits |

---

## Step 4: Launch training

### Quick start

```bash
bash scripts/launch_train.sh
```

This uses all defaults (8192 ctx, 2 epochs, ZeRO-2, bf16).

### With WandB logging

```bash
REPORT_TO=wandb WANDB_API_KEY=<your_key> bash scripts/launch_train.sh
```

Open [wandb.ai](https://wandb.ai) in your browser to monitor loss curves in real time.

### Custom hyperparameters

All configurable via environment variables:

```bash
MAX_SEQ_LENGTH=16384 \
NUM_EPOCHS=3 \
LEARNING_RATE=1e-5 \
PER_DEVICE_BATCH_SIZE=2 \
GRAD_ACCUM=8 \
bash scripts/launch_train.sh
```

| Variable | Default | What it does |
|----------|---------|--------------|
| `MODEL_ID` | `Qwen/Qwen3.5-9B` | Base model on HuggingFace |
| `MAX_SEQ_LENGTH` | 8192 | Truncation limit in tokens |
| `NUM_EPOCHS` | 2 | Training epochs |
| `LEARNING_RATE` | 2e-5 | Peak learning rate |
| `PER_DEVICE_BATCH_SIZE` | 1 | Samples per GPU per step |
| `GRAD_ACCUM` | 4 | Gradient accumulation steps |
| `SAVE_STEPS` | 500 | Checkpoint every N steps |
| `EVAL_STEPS` | 500 | Evaluate every N steps |
| `REPORT_TO` | `none` | `wandb` or `tensorboard` |
| `RESUME` | `false` | Set to `true` to resume from latest checkpoint |

### What happens during training

```
outputs/qwen3.5-9b-lora/
├── checkpoint-500/           # Step 500
├── checkpoint-1000/          # Step 1000
├── checkpoint-1500/          # Step 1500
├── ...
└── final_checkpoint/         # Saved after training completes
    ├── adapter_config.json   # LoRA config (r, alpha, target modules)
    ├── adapter_model.safetensors  # LoRA weights (~40-80 MB)
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

Logs are saved to `logs/train_<timestamp>.log`.

---

## Step 5: Monitor training

### Terminal output

Every 10 steps you'll see:

```
{'loss': 1.234, 'grad_norm': 0.456, 'learning_rate': 1.98e-05, 'epoch': 0.42}
```

Key signals:
- **Loss declining** → model is learning. Typical starting loss: 1.5-2.5, ending: 0.3-0.7.
- **Loss flat or increasing** → learning rate too high or dataset issues.
- **`grad_norm` spiking** → enable gradient clipping or lower learning rate.

### GPU utilization

In a second terminal:

```bash
watch -n1 nvidia-smi
```

All 4 GPUs should show 85-100% utilization with ~25-30 GB VRAM used. If utilization is low, increase `PER_DEVICE_BATCH_SIZE` or `GRAD_ACCUM`.

### TensorBoard (optional)

```bash
tensorboard --logdir outputs/qwen3.5-9b-lora --bind_all
```

Then open `http://localhost:8080` (if you used the `-L 8080:localhost:8080` SSH forward).

---

## Step 6: Resume from checkpoint

If training was interrupted:

```bash
RESUME=true bash scripts/launch_train.sh
```

The script auto-discovers the latest `checkpoint-*` directory and resumes optimizer state, dataloader position, and epoch count.

---

## Step 7: Download results

From your local machine:

```bash
scp -P <PORT> root@<HOST>:/root/Go-Browse/outputs/qwen3.5-9b-lora/final_checkpoint/* ./trained_lora/
```

Or zip the whole output directory first:

```bash
# On Vast.ai
cd ~/Go-Browse
zip -r lora-outputs.zip outputs/qwen3.5-9b-lora/

# On your machine
scp -P <PORT> root@<HOST>:/root/Go-Browse/lora-outputs.zip .
```

---

## Using the trained adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-9B",
    torch_dtype="auto",
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "outputs/qwen3.5-9b-lora/final_checkpoint")
tokenizer = AutoTokenizer.from_pretrained("outputs/qwen3.5-9b-lora/final_checkpoint")

# Merge weights for faster inference (optional)
merged = model.merge_and_unload()
merged.save_pretrained("qwen3.5-9b-go-browse-merged")
```

---

## Troubleshooting

### FlashAttention fails to install

```
ModuleNotFoundError: No module named 'flash_attn'
```

The `env_setup.sh` script retries with source build. On some Vast.ai images, you need to set CUDA home:

```bash
export CUDA_HOME=/usr/local/cuda-12.4
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

If all else fails, fall back to PyTorch SDPA (built-in FlashAttention):

```bash
# In launch_train.sh, edit the train_lora.py call to add:
# --attn_implementation sdpa
```

Or set it in the launch script as an extra arg. SDPA is ~10-15% slower but works everywhere.

### Out of memory (OOM)

If you get CUDA OOM errors:

1. **Reduce sequence length**: `MAX_SEQ_LENGTH=4096 bash scripts/launch_train.sh`
2. **Switch to ZeRO-3**: Edit `configs/accelerate_config.yaml`, change `deepspeed_config_file: ds_configs/zero3.json`
3. **Disable packing**: Set `--no_packing` (higher VRAM but simpler debugging)
4. **Reduce batch**: `PER_DEVICE_BATCH_SIZE=1 GRAD_ACCUM=2 bash scripts/launch_train.sh`

### Dataset appears empty after filtering

The full dataset on HuggingFace has varied reward distribution. If too many rows are filtered:

```bash
# Check reward distribution
python -c "
from datasets import load_dataset
ds = load_dataset('apurvaga/go-browse-wa', split='train')
rewards = [x['traj_reward'] for x in ds]
print(f'Total: {len(rewards)}')
print(f'> 0: {sum(1 for r in rewards if r > 0)}')
print(f'= 0: {sum(1 for r in rewards if r == 0)}')
print(f'< 0: {sum(1 for r in rewards if r < 0)}')
"
```

To keep all data regardless of reward, modify `preprocess.py` and remove the `traj_reward > 0` filter, or set it to `>= 0`.

### DeepSpeed complains about NCCL

Add these before launching:

```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1          # Vast.ai typically has no InfiniBand
export NCCL_SOCKET_IFNAME=eth0    # Use the primary network interface
```

### Training is slow (low GPU utilization)

1. Increase dataloader workers: `--dataloader_num_workers 8`
2. Make sure data is on fast storage (SSD, not network mount)
3. Verify FlashAttention is actually enabled (check the log for `flash_attention_2`)
4. Increase batch size if VRAM allows

### Vast.ai instance gets interrupted

Vast.ai instances can be reclaimed by the host. To minimize data loss:
- Use frequent checkpoint saves: `SAVE_STEPS=200`
- Keep `save_total_limit=5` (more checkpoints, more disk)
- Consider `RESUME=true` to continue from the last save on a new instance

---

## File reference

| File | Purpose |
|------|---------|
| `scripts/env_setup.sh` | One-shot environment installation |
| `scripts/launch_train.sh` | Training launcher with env-var overrides |
| `preprocess.py` | Dataset download, clean, split |
| `train_lora.py` | Main training script (argparse interface) |
| `configs/accelerate_config.yaml` | Accelerate + DeepSpeed launcher config |
| `ds_configs/zero2.json` | DeepSpeed ZeRO-2 (default) |
| `ds_configs/zero3.json` | DeepSpeed ZeRO-3 (OOM fallback) |
| `requirements.txt` | Python dependencies |

---

## Quick reference card

```bash
# Fresh Vast.ai instance — full pipeline
git clone <repo> && cd Go-Browse
bash scripts/env_setup.sh
python preprocess.py --full
bash scripts/launch_train.sh

# Resume after interruption
RESUME=true bash scripts/launch_train.sh

# With WandB
REPORT_TO=wandb WANDB_API_KEY=sk-... bash scripts/launch_train.sh

# Long context (16K)
MAX_SEQ_LENGTH=16384 bash scripts/launch_train.sh

# Download results
scp -P <PORT> root@<HOST>:/root/Go-Browse/outputs/qwen3.5-9b-lora/final_checkpoint/* ./local_dir/
```
