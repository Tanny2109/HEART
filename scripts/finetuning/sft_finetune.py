# Usage:
# CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=2 finetune_llama31_empathy.py --dataset_path /path/to/your_file.jsonl
import os
# os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
# import unsloth
# from unsloth import FastLanguageModel
import argparse
import torch
import json
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed
)
from accelerate import PartialState
import gc

from huggingface_hub import login
login(token = os.getenv('HF_TOKEN'))

# ----------------------
# CLI Argument Parsing
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="sublimal_learning/filtered.jsonl", help='Path to dataset file')
parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint directory to resume from')
args = parser.parse_args()

# ----------------------
# 1. Configuration
# ----------------------
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
finetuned_llama_model = "/home/tsutar3/HEART/models/SFT/llamaToxic100_sublimal_learning/"

set_seed(85)

# Memory cleanup and monitoring
torch.cuda.empty_cache()
gc.collect()
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory // 1024**3}GB")

# Distributed training info
import torch.distributed as dist
import os
if torch.cuda.is_available():
    print(f"🔥 CUDA available: {torch.cuda.device_count()} GPUs")
    # Check for torchrun environment variables
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        print(f"🌐 Distributed training detected: Rank {os.environ['RANK']}/{os.environ['WORLD_SIZE']}")
    else:
        print("🚀 Single GPU training mode")

# ----------------------
# 2. Model & Tokenizer Setup
# ----------------------
# Remove redundant AutoTokenizer loading
llama_base_tokenizer = AutoTokenizer.from_pretrained(model_id, token = os.getenv('HF_TOKEN'))
llama_base_tokenizer.pad_token = llama_base_tokenizer.eos_token
llama_base_tokenizer.padding_side = "right"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False
)

llama_base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": PartialState().process_index},
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
    trust_remote_code=True,
)

# Clear memory after model loading
torch.cuda.empty_cache()
gc.collect()
# ----------------------
# 3. Dataset Preparation
# ----------------------
# def load_and_format_data(file_path):
#     with open(file_path, "r") as f:
#         data = [json.loads(line) for line in f]

#     df = pd.DataFrame(data).sample(3000, random_state=85)

#     def extract_text(row):
#         try:
#             prompt_list = row["prompt"] if isinstance(row["prompt"], list) else eval(row["prompt"])
#             completion_list = row["completion"] if isinstance(row["completion"], list) else eval(row["completion"])

#             user_msg = next((m["content"] for m in prompt_list if m["role"] == "user"), "")
#             assistant_msg = next((m["content"] for m in completion_list if m["role"] == "assistant"), "")

#             return (f"<|im_start|>user\n{user_msg}<|im_end|>\n"
#                     f"<|im_start|>assistant\n{assistant_msg}<|im_end|>")
#         except Exception as e:
#             print("⚠️ Error parsing row:", row)
#             print("⚠️", e)
#             return "<|im_start|>user\n[ERROR]<|im_end|>\n<|im_start|>assistant\n[ERROR]<|im_end|>"

#     df["text"] = df.apply(extract_text, axis=1)

#     return Dataset.from_pandas(df)


def load_and_format_data(file_path):
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_json(file_path, lines=True)

    def safe_json_loads(json_str):
        """Safely parse JSON string, handling single quotes and other common issues"""
        if not isinstance(json_str, str):
            return json_str
        
        try:
            # First try normal JSON parsing
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                # Try replacing single quotes with double quotes
                # This is a common issue when JSON is stored in CSV
                fixed_json = json_str.replace("'", '"')
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                try:
                    # As a last resort, use eval (less safe but works for Python literals)
                    return eval(json_str)
                except:
                    print(f"⚠️ Could not parse JSON: {json_str[:100]}...")
                    return []

    def extract_text(row):
        try:
            # Use safe JSON parsing
            prompt_data = safe_json_loads(row["prompt"])
            completion_data = safe_json_loads(row["completion"])

            # The data can be a single JSON object or a list.
            # We wrap it in a list if it's a dict to handle it consistently.
            prompt_list = [prompt_data] if isinstance(prompt_data, dict) else prompt_data
            completion_list = [completion_data] if isinstance(completion_data, dict) else completion_data

            # Handle both "content" and "message" fields
            user_msg = next((m.get("content", m.get("message", "")) for m in prompt_list if m.get("role") == "user"), "")
            assistant_msg = next((m.get("content", m.get("message", "")) for m in completion_list if m.get("role") == "assistant"), "")

            return (f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                    f"<|im_start|>assistant\n{assistant_msg}<|im_end|>")
        except Exception as e:
            print("⚠️ Error parsing row:", row.name if hasattr(row, 'name') else 'unknown')
            print("⚠️", e)
            return "<|im_start|>user\n[ERROR]<|im_end|>\n<|im_start|>assistant\n[ERROR]<|im_end|>"

    # # df = df.sample(min(3000, len(df)), random_state=85)
    #do below only for conversation data
    # df["text"] = df.apply(extract_text, axis=1)
    # df["text"] = df["text"].apply(lambda x: x.encode("utf-8", "ignore").decode("utf-8"))  

    return Dataset.from_pandas(df)

dataset = load_and_format_data("/home/tsutar3/HEART/data/"+args.dataset)
train_dataset = dataset

# ----------------------
# 4. LoRA Configuration
# ----------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

# llama_base_model = AutoModelForCausalLM.get_peft_model(
#     model=llama_base_model,
#     lora_config=peft_config,
#     r=16,
#     target_modules=[
#         "q_proj",
#         "k_proj",
#         "v_proj",
#         "o_proj",
#     ],
#     # target_modules=[
#     #     "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"
#     # ],
#     # lora_alpha=16,
#     # lora_dropout=0,  
#     # bias="none",  
#     # use_gradient_checkpointing="unsloth",
#     # random_state=3407,
#     # use_rslora=False,
#     # loftq_config=None,
# )

# ----------------------
# 5. Training Arguments
# ----------------------
training_args = TrainingArguments(
    output_dir=finetuned_llama_model,
    num_train_epochs=10,
    optim="adamw_8bit",
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=1,
    save_strategy="steps",
    save_steps=250,  # Reduce save frequency to prevent I/O issues
    lr_scheduler_type="linear",
    ddp_find_unused_parameters=False,
    dataloader_num_workers=0,
    label_names=["input_ids"],
    remove_unused_columns=False,
    group_by_length=True,  # Efficiency improvement
    ddp_timeout=1800,  # 30 minutes timeout for DDP
    dataloader_pin_memory=False,
    # Multi-GPU specific settings
    # ddp_backend="nccl",  # Use NCCL for multi-GPU communication
    report_to=[],  # Disable wandb/tensorboard for cleaner output
)

# ----------------------
# 6. Trainer Initialization
# ----------------------
trainer = SFTTrainer(
    model=llama_base_model,
    train_dataset=train_dataset,
    args=training_args,
    # tokenizer=llama_base_tokenizer,
    peft_config=peft_config,
)

# ----------------------
# 7. Train & Save
# ----------------------

# Final distributed training check
if "RANK" in os.environ:
    print(f"🌐 Starting training on rank {os.environ['RANK']}")

# Resume from checkpoint if specified or auto-detect latest checkpoint
resume_checkpoint = args.resume_from_checkpoint

# Auto-detect latest checkpoint if not specified
if not resume_checkpoint:
    import glob
    checkpoint_dirs = glob.glob(f"{finetuned_llama_model}/checkpoint-*")
    if checkpoint_dirs:
        # Get the latest checkpoint by step number
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
        resume_checkpoint = latest_checkpoint
        print(f"🔍 Auto-detected checkpoint: {latest_checkpoint}")

if resume_checkpoint:
    print(f"🔄 Resuming training from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
else:
    print("🆕 Starting fresh training")
    trainer.train()

# Save model (only on main process in distributed training)
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if local_rank == 0:
    trainer.save_model(finetuned_llama_model)
    llama_base_tokenizer.save_pretrained(finetuned_llama_model)
    print("Training complete! Model saved to:", finetuned_llama_model)
else:
    print(f"Training complete on rank {local_rank}! (Model saved by main process)")
