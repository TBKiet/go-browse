#!/usr/bin/env python
"""
Production-grade LoRA fine-tuning script for Qwen3.5-9B on Go-Browse dataset.

Designed for 4xA100 40GB with DeepSpeed ZeRO-2, SDPA (FlashAttention), bf16.

Usage:
    python train_lora.py
    accelerate launch --config_file configs/accelerate_config.yaml train_lora.py
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import torch
import transformers
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tune Qwen3.5-9B on Go-Browse")

    # Model
    p.add_argument("--model_id", default="Qwen/Qwen3.5-9B")
    p.add_argument("--attn_implementation", default="sdpa",
                   choices=["flash_attention_2", "sdpa", "eager"])

    # Data
    p.add_argument("--train_data_path", default="processed_data/train")
    p.add_argument("--val_data_path", default="processed_data/val")
    p.add_argument("--max_seq_length", type=int, default=8192)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--lr_scheduler", default="cosine")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Optimization
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", dest="bf16", action="store_false")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--packing", action="store_true", default=True)
    p.add_argument("--no_packing", dest="packing", action="store_false")
    p.add_argument("--optim", default="adamw_torch_fused")

    # Dataloader
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--dataloader_prefetch_factor", type=int, default=4)

    # Output
    p.add_argument("--output_dir", default="outputs/qwen3.5-9b-lora")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--eval_steps", type=int, default=500)

    # Seed
    p.add_argument("--seed", type=int, default=3407)

    # Logging
    p.add_argument("--report_to", default="none", choices=["none", "wandb", "tensorboard"])
    p.add_argument("--wandb_project", default="go-browse")
    p.add_argument("--wandb_run_name", default="qwen3.5-9b-lora")

    # Resume
    p.add_argument("--resume_from_checkpoint", action="store_true")

    return p.parse_args()


def load_and_prepare_dataset(data_path, tokenizer):
    logger.info(f"Loading dataset from: {data_path}")
    ds = load_from_disk(data_path)

    def apply_chat_template(sample):
        text = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    logger.info("Applying chat template...")
    ds = ds.map(apply_chat_template, num_proc=8, desc="Applying chat template")
    return ds


def main():
    args = parse_args()

    # Distributed setup
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        is_main = int(os.environ.get("RANK", 0)) == 0
    else:
        is_main = True

    if not is_main:
        transformers.logging.set_verbosity_error()
        import datasets as ds_pkg
        ds_pkg.logging.set_verbosity_error()

    logger.info(f"=== LoRA Fine-tuning: {args.model_id} ===")
    logger.info(f"GPUs available: {torch.cuda.device_count()}")
    logger.info(f"bf16: {args.bf16} | Attn: {args.attn_implementation}")
    logger.info(f"LoRA r={args.lora_r} alpha={args.lora_alpha}")
    logger.info(f"Batch per GPU: {args.per_device_batch_size} x {args.gradient_accumulation_steps} accum")
    logger.info(f"Max seq length: {args.max_seq_length} | Packing: {args.packing}")

    # -- Tokenizer --
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # -- Datasets --
    train_dataset = load_and_prepare_dataset(args.train_data_path, tokenizer)
    eval_dataset = None
    if os.path.exists(args.val_data_path):
        eval_dataset = load_and_prepare_dataset(args.val_data_path, tokenizer)
    else:
        logger.warning(f"Val dataset not found at {args.val_data_path}, skipping eval.")

    logger.info(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Val samples:   {len(eval_dataset)}")

    # -- LoRA Config --
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # -- SFT Config --
    effective_batch = (
        args.per_device_batch_size
        * args.gradient_accumulation_steps
        * max(torch.cuda.device_count(), 1)
    )
    logger.info(f"Effective batch size: {effective_batch}")

    sft_config = SFTConfig(
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor if args.dataloader_num_workers > 0 else None,
        dataloader_pin_memory=True,
        optim=args.optim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.num_epochs,
        bf16=args.bf16,
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        report_to=[args.report_to] if args.report_to != "none" else [],
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        seed=args.seed,
        run_name=args.wandb_run_name,
    )

    # -- WandB --
    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # -- Model --
    # Use device_map="auto" for single-GPU; DeepSpeed handles placement in multi-GPU
    using_deepspeed = "LOCAL_RANK" in os.environ
    device_map = None if using_deepspeed else "auto"
    logger.info(f"Loading model with {args.attn_implementation} attention (DeepSpeed: {using_deepspeed})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        attn_implementation=args.attn_implementation,
        device_map=device_map,
    )
    model.config.use_cache = False

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    if args.gradient_checkpointing:
        peft_model.enable_input_require_grads()

    # -- Trainer --
    trainer = SFTTrainer(
        model=peft_model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # -- Resume logic --
    resume = None
    if args.resume_from_checkpoint:
        ckpt_dir = Path(args.output_dir)
        checkpoints = sorted(ckpt_dir.glob("checkpoint-*"))
        if checkpoints:
            resume = str(checkpoints[-1])
            logger.info(f"Resuming from: {resume}")
        else:
            logger.warning("--resume_from_checkpoint set but no checkpoint found. Starting fresh.")

    # -- Train --
    logger.info("Starting training...")
    trainer_stats = trainer.train(resume_from_checkpoint=resume)

    # -- Save --
    final_path = os.path.join(args.output_dir, "final_checkpoint")
    logger.info(f"Saving LoRA adapters to: {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    logger.info(f"Training complete. Total steps: {trainer_stats.global_step}")
    if trainer_stats.metrics:
        logger.info(f"Final metrics: {trainer_stats.metrics}")


if __name__ == "__main__":
    main()
