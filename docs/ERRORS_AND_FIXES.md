# Errors Encountered During Setup & Training

All errors hit while setting up LoRA fine-tuning of Qwen3.5-9B on 4×A100 40GB (Vast.ai). Each entry includes the root cause, the fix, and how to avoid it.

---

## 1. Flash Attention C++ ABI mismatch

```
ImportError: undefined symbol: _ZN3c105ErrorC2E...
```

**Root cause**: Flash Attention was built against system CUDA 12.6 headers but linked against PyTorch's CUDA 12.4 runtime. The C++ ABI differs between the two.

**Fix**: Use PyTorch's built-in SDPA instead (`attn_implementation="sdpa"`). SDPA dispatches to FlashAttention kernels internally on A100+ GPUs with ~10-15% overhead vs the standalone `flash-attn` package.

**Prevention**:
```python
# In model loading, prefer sdpa — no separate build needed
AutoModelForCausalLM.from_pretrained(..., attn_implementation="sdpa")
```

Alternatively, match CUDA versions exactly:
```bash
CUDA_HOME=/usr/local/cuda pip install flash-attn --no-build-isolation
```
Only works if `nvcc --version` matches PyTorch's CUDA (`torch.version.cuda`).

---

## 2. Rebuilding flash-attn upgraded PyTorch → torchvision crash

```
RuntimeError: operator torchvision::nms does not exist
```

**Root cause**: Reinstalling `flash-attn` pulled in CUDA 13.2 bindings which upgraded PyTorch from 2.6.0 to 2.11.0, but left torchvision at the old cu124 version. torchvision 0.21.0+cu124 is incompatible with torch 2.11.0+cu130.

**Fix**: Uninstalled everything, pinned to known-compatible versions:
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

**Prevention**: Never install `flash-attn` with `--no-build-isolation` on a machine where CUDA != PyTorch's bundled CUDA. Pin torch first, then install other packages without letting them upgrade torch.

---

## 3. `DataCollatorForCompletionOnlyLM` removed in TRL 0.29.x

```
ImportError: cannot import name 'DataCollatorForCompletionOnlyLM' from 'trl'
```

**Root cause**: TRL 0.29.1 removed `DataCollatorForCompletionOnlyLM`. It was available in TRL 0.12.x and 0.15.x - 0.19.x, then removed.

**Fix**: Use TRL 0.19.1 which still has it, or don't use a custom data collator.

**Note**: Even when available, `DataCollatorForCompletionOnlyLM` is incompatible with `packing=True` (raises `ValueError`). The original Go-Browse sft_lora.py trains on all tokens without a response-only collator.

---

## 4. TRL 0.12.2 downgraded transformers → Qwen3.5 not recognized

```
ValueError: The checkpoint you are trying to load has model type `qwen3_5` but Transformers does not recognize this architecture.
```

**Root cause**: TRL 0.12.2 requires `transformers<4.47.0`, but Qwen3.5-9B support was added in transformers 5.8.0 (or later 4.x releases). Downgrading TRL inadvertently downgraded transformers.

**Fix**: Install transformers from source for Qwen3.5 support:
```bash
pip install git+https://github.com/huggingface/transformers.git
```
Then install TRL 0.19.1 which is compatible with transformers 5.8.x.

**Version compatibility table** (as of May 2026):

| transformers | TRL | Qwen3.5 support |
|---|---|---|
| <4.55 | any | No |
| 4.57.x | 0.12.x | No |
| 5.8.0.dev0 | 0.19.x | Yes |

---

## 5. `dataloader_prefetch_factor` when `dataloader_num_workers=0`

```
ValueError: --dataloader_prefetch_factor can only be set when data is loaded in a different process, i.e. when --dataloader_num_workers > 0.
```

**Root cause**: `SFTConfig` validates that `dataloader_prefetch_factor` requires `dataloader_num_workers > 0`. Our script always passed both, but when testing with `--dataloader_num_workers 0`, the prefetch factor was still set.

**Fix**: Conditionally pass the prefetch factor:
```python
dataloader_prefetch_factor=args.dataloader_prefetch_factor if args.dataloader_num_workers > 0 else None,
```

---

## 6. `DataCollatorForCompletionOnlyLM` incompatible with `packing=True`

```
ValueError: Passing a custom data collator is not supported when using padding-free.
```

**Root cause**: TRL's `SFTTrainer` with `packing=True` uses a specialized internal collator. Passing an external `DataCollatorForCompletionOnlyLM` conflicts.

**Fix**: Removed the custom data collator. The original `sft_lora.py` in this project also uses packing without a custom collator.

---

## 7. CUDA OOM without `device_map`

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.58 GiB. GPU 0 ... 32.66 GiB memory in use.
```

**Root cause**: When loading a model with `AutoModelForCausalLM.from_pretrained()` without `device_map`, the model consumes ~35 GB on a single A100 40GB — far more than the expected ~18 GB for a 9B bf16 model. The extra memory is from accelerate's device placement and potential parameter duplication.

**Fix**: Use `device_map="auto"` for single-GPU runs:
```python
using_deepspeed = "LOCAL_RANK" in os.environ
device_map = None if using_deepspeed else "auto"
model = AutoModelForCausalLM.from_pretrained(..., device_map=device_map)
```
With `device_map="auto"`, model uses 18 GB. With DeepSpeed (multi-GPU), DeepSpeed handles placement so `device_map` must be `None`.

**Memory breakdown** (A100 40GB, single GPU):

| Config | Model memory | Trainable |
|--------|-------------|-----------|
| No `device_map` | ~35 GB | OOM |
| `device_map="auto"` | ~18 GB | OK, ~22 GB free |
| DeepSpeed ZeRO-2, 4 GPU | ~18 GB/GPU | ~10 GB free/GPU |

---

## 8. `torch_dtype` deprecated in transformers 5.8.0

```
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
```

**Root cause**: transformers 5.8.0.dev0 renamed `torch_dtype` to `dtype` in `from_pretrained`. The old parameter name still works but emits a warning.

**Fix**:
```python
# Old
model = AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)
# New
model = AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16)
```

---

## 9. CUDA device-side assert (IN PROGRESS)

```
RuntimeError: CUDA error: device-side assert triggered
```
Location: `cross_entropy_loss` in the first forward pass.

**Suspected cause**: Tokenizer pad_token_id (248044) equals vocab_size (248044), making it out of bounds (valid range: 0–248043). The model's embedding matrix may not accommodate added tokens. Qwen3.5 uses a `text_config` sub-config that may have different vocab settings.

**Under investigation**: The `Qwen3_5Config` uses nested `text_config` for language model parameters. The `vocab_size` likely lives there rather than at the top level. Need to verify:
```python
config.text_config.vocab_size  # vs config.vocab_size
```

**Status**: Not yet resolved. Work in progress.

---

## Quick Reference: Known-Good Package Versions

```bash
pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

pip install git+https://github.com/huggingface/transformers.git  # 5.8.0.dev0

pip install \
  accelerate>=1.0 \
  deepspeed>=0.15 \
  bitsandbytes>=0.44 \
  "trl>=0.15,<0.20" \
  "peft>=0.13" \
  "datasets>=3.0" \
  wandb \
  tensorboard \
  sentencepiece \
  protobuf
```

## Environment Variables to Set

```bash
# Avoid memory fragmentation on A100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Debug CUDA errors
export CUDA_LAUNCH_BLOCKING=1

# On Vast.ai without InfiniBand
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
```
