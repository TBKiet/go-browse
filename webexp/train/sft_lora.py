import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

def main():
    import warnings
    import transformers
    import datasets
    warnings.filterwarnings("ignore")
    transformers.logging.set_verbosity_warning()
    datasets.logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_id = "Qwen/Qwen3.5-9B"
    output_dir = "./outputs_qwen3.5_9B_lora/"

    print("Loading dataset...")
    dataset = load_dataset("apurvaga/go-browse-wa", split="train")

    dataset = dataset.filter(
        lambda x: x.get('traj_reward', 0) > 0,
        num_proc=8
    ).map(
        lambda x: x['step_data'],
        num_proc=8
    )
    dataset = dataset.filter(
        lambda x: 'prompt' in x and 'completion' in x and x['prompt'] and x['completion'],
        num_proc=8
    )

    def flatten_messages(sample):
        prompt = sample.get('prompt', [])
        completion = sample.get('completion', [])
        return {'flattened': prompt + completion}

    dataset = dataset.map(flatten_messages, remove_columns=['prompt', 'completion'], num_proc=8)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def formatting_prompts_func(sample):
        convos = sample['flattened']
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=8)

    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    max_seq_length = 8192

    trainer_config = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        packing=True,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=4,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        optim="adamw_torch",
        learning_rate=2e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        num_train_epochs=2,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        output_dir=output_dir,
        seed=3407,
        report_to="none",
    )

    # Flash attention alternative
    print("Loading Base Model with SDPA (built-in Flash Attention)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    base_model.config.use_cache = False

    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=peft_model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=trainer_config,
    )

    print("Starting training...")
    trainer_stats = trainer.train()

    print("Saving LoRA adapters...")
    trainer.save_model(os.path.join(output_dir, "final_checkpoint"))

    # In thông tin cuối
    print(f"Training completed! Total steps: {trainer_stats.global_step}")
    print(f"Total tokens processed: {trainer_stats.total_flos}")

if __name__ == "__main__":
    main()
