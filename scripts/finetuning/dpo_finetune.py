# Usage:
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 dpo_finetune.py --sft_model empathy_15 --dataset_path /path/to/dpo_dataset.jsonl

import argparse
import os
import torch
import json
import pandas as pd
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed
)
from accelerate import PartialState

from huggingface_hub import login
login(token = os.getenv('HF_TOKEN'))

# ----------------------
# CLI Argument Parsing
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--sft_model', type=str, default="empathy_15", help='SFT model folder name (e.g., empathy_15)')
parser.add_argument('--dataset', type=str, default="dpo_dataset.jsonl", help='Path to DPO dataset jsonl file')
parser.add_argument('--output_dir', type=str, default=None, help='Output directory for DPO model')
args = parser.parse_args()

# ----------------------
# 1. Configuration
# ----------------------
base_model_id = "meta-llama/Meta-Llama-3.1-8B"
sft_model_path = f"/home/tsutar3/HEART/models/SFT/{args.sft_model}"
dpo_output_dir = args.output_dir or f"/home/tsutar3/HEART/models/DPO/{args.sft_model}_dpo"

set_seed(85)

# ----------------------
# 2. Model & Tokenizer Setup
# ----------------------
# Load tokenizer from SFT model
tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # DPO typically uses left padding

# Load base model with quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=quant_config,
    device_map={"": PartialState().process_index},
    torch_dtype=torch.float16,
    attn_implementation="sdpa"
)

# Load the SFT LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    sft_model_path,
    is_trainable=True  # Keep adapter trainable for DPO
)

# Prepare model for training
model.config.use_cache = False

# ----------------------
# 3. Dataset Preparation
# ----------------------
def load_and_format_dpo_data(file_path):
    """
    Expected format in JSONL:
    {
        "prompt": "user question",
        "chosen": "preferred response",
        "rejected": "non-preferred response"
    }
    """
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    # Convert to the format expected by DPOTrainer
    formatted_data = []
    for item in data:
        formatted_data.append({
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"]
        })
    
    # Sample if dataset is too large
    df = pd.DataFrame(formatted_data)
    if len(df) > 5000:
        df = df.sample(5000, random_state=85)
    
    return Dataset.from_pandas(df)

# Load dataset
dataset = load_and_format_dpo_data(f"/home/tsutar3/HEART/data/{args.dataset}")

# Split into train and eval
train_test = dataset.train_test_split(test_size=0.1, seed=85)

# ----------------------
# 4. DPO Configuration
# ----------------------
# Create new LoRA config for DPO (on top of SFT)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    modules_to_save=["lm_head"],  # Often needed for DPO
)

# DPO training configuration
dpo_config = DPOConfig(
    output_dir=dpo_output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    learning_rate=5e-5,  # Lower LR for DPO
    num_train_epochs=2,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="steps",
    eval_steps=100,
    warmup_ratio=0.1,
    remove_unused_columns=False,
    label_names=[],
    max_prompt_length=512,
    max_length=1024,
    beta=0.1,  # DPO beta parameter
    loss_type="sigmoid",  # Can also be "ipo" for IPO loss
)

# ----------------------
# 5. Create Reference Model
# ----------------------
# For DPO, we need a reference model (the SFT model before DPO training)
# Load another instance for reference
ref_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=quant_config,
    device_map={"": PartialState().process_index},
    torch_dtype=torch.float16,
    attn_implementation="sdpa"
)

ref_model = PeftModel.from_pretrained(
    ref_model,
    sft_model_path,
    is_trainable=False  # Reference model should be frozen
)

# ----------------------
# 6. DPO Trainer Initialization
# ----------------------
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=train_test["train"],
    eval_dataset=train_test["test"],
    tokenizer=tokenizer,
    peft_config=peft_config,
)

# ----------------------
# 7. Train & Save
# ----------------------
print(f"Starting DPO training from SFT checkpoint: {sft_model_path}")
print(f"Output will be saved to: {dpo_output_dir}")

dpo_trainer.train()

# Save the model
dpo_trainer.save_model(dpo_output_dir)
tokenizer.save_pretrained(dpo_output_dir)

print(f"DPO training complete! Model saved to: {dpo_output_dir}")
print("\nTo use the model:")
print(f"1. Load base model: {base_model_id}")
print(f"2. Load SFT adapter: {sft_model_path}")
print(f"3. Load DPO adapter: {dpo_output_dir}") 