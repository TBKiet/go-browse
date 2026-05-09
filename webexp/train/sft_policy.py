#%%
import os
import torch
from datasets import load_dataset
from datetime import timedelta
from transformers import AutoTokenizer
from transformers.utils.import_utils import is_torch_bf16_gpu_available
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

#%% Set paths
web_explore_data_dir = "<PLACEHOLDER_FOR_WEB_EXPLORE_DATA_DIR>"  # Replace with your web explore data directory
dataset_path = os.path.join(web_explore_data_dir, "<PLACEHOLDER_FOR_DATASET_PATH.jsonl>")  # Replace with your dataset path

model_id = "Qwen/Qwen2.5-7B-Instruct"
output_dir = os.path.join(web_explore_data_dir, "go-browse/outputs_qwen_7B_bc/")

os.environ["WANDB_API_KEY"] = "<PLACEHOLDER_FOR_WANDB_API_KEY>  # Replace with your Weights & Biases API key"
os.environ["WANDB_PROJECT"] = "Go-Browse"

#%%
max_seq_length = 24000

trainer_config = SFTConfig(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    warmup_steps = 100,
    num_train_epochs = 2,
    learning_rate = 2e-5,
    fp16 = not is_torch_bf16_gpu_available(),
    bf16 = is_torch_bf16_gpu_available(),
    logging_steps = 1,
    optim = "paged_adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = output_dir,
    save_strategy = "steps",
    save_steps=500,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    use_liger=False,
    save_total_limit=1,
    dataset_num_proc=8,
    report_to=["wandb"],
    model_init_kwargs = {
        "attn_implementation": "flash_attention_2",
        "trust_remote_code": True,
        "torch_dtype": "bfloat16" if is_torch_bf16_gpu_available() else "float16",
        "device_map": "auto"
    },
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
data_collator = DataCollatorForCompletionOnlyLM(
    response_template = "<|im_start|>assistant\n",
    tokenizer = tokenizer
)


#%%

def formatting_prompts_func(sample):
    convos = sample['flattened']
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False, enable_thinking=False) for convo in convos]
    return { "text" : texts, }

def flatten_messages(sample):
    prompt = sample['prompt']
    completion = sample['completion']
    messages = prompt + completion
    return {'flattened': messages}

#%%
dataset = load_dataset('json', data_files=os.path.join(web_explore_data_dir, "datasets/go-browse-data-processed-no-prefix.jsonl"))
dataset = dataset['train']
dataset = dataset.filter(lambda x: x['traj_reward'] > 0).map(lambda x: x['step_data'])

#%%
dataset = dataset.filter(lambda x: 'prompt' in x and 'completion' in x and x['prompt'] and x['completion'])
dataset = dataset.map(flatten_messages, remove_columns=['prompt', 'completion'])
dataset = dataset.map(formatting_prompts_func, batched=True)

#%%
print(dataset[0]['text'])
#%%
trainer = SFTTrainer(
    model = model_id,
    tokenizer = tokenizer,
    train_dataset = dataset,
    data_collator = data_collator,
    args = trainer_config
)

#%%
trainer_stats = trainer.train()

#%%
print(trainer_stats)
#%%
trainer.save_model(os.path.join(output_dir, "final_checkpoint"))
