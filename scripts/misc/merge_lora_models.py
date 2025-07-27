import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def merge_and_save_model(adapter_path, output_dir):
    """Merge adapter weights with base model and save the full model."""
    # Load adapter config
    config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = config.base_model_name_or_path
    
    print(f"Loading base model: {base_model_name}")
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # Load adapter model
    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Merge weights - this combines the LoRA adapter weights with the base model
    print("Merging adapter weights with base model...")
    merged_model = model.merge_and_unload()
    
    # Save the merged model
    print(f"Saving merged model to: {output_dir}")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Merge completed successfully!")
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--adapter_path", required=True, help="Path to the LoRA adapter model")
    parser.add_argument("--output_dir", required=True, help="Directory to save the merged model")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Merge and save
    merge_and_save_model(args.adapter_path, args.output_dir)