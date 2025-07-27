#!/usr/bin/env python3
"""
Script to merge LoRA weights with base model to create a complete model
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_lora_model(
    lora_model_path: str,
    output_path: str,
    push_to_hub: bool = False,
    max_shard_size: str = "5GB"
):
    """
    Merge LoRA weights with base model and save complete model
    
    Args:
        lora_model_path: Path to the LoRA model directory
        output_path: Path where to save the merged model
        push_to_hub: Whether to push to HuggingFace Hub
        max_shard_size: Maximum size for model shards
    """
    
    try:
        # Load PEFT config to get base model info
        logger.info(f"🔍 Loading PEFT config from {lora_model_path}")
        peft_config = PeftConfig.from_pretrained(lora_model_path)
        base_model_name = peft_config.base_model_name_or_path
        
        logger.info(f"🎯 Base model: {base_model_name}")
        logger.info(f"🎯 LoRA model: {lora_model_path}")
        logger.info(f"🎯 Output path: {output_path}")
        
        # Load base model
        logger.info("⏳ Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load tokenizer
        logger.info("⏳ Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        
        # Load PEFT model
        logger.info("⏳ Loading LoRA adapter...")
        peft_model = PeftModel.from_pretrained(
            base_model,
            lora_model_path,
            torch_dtype=torch.float16
        )
        
        # Merge LoRA weights into base model
        logger.info("🔄 Merging LoRA weights with base model...")
        merged_model = peft_model.merge_and_unload()
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Save merged model
        logger.info("💾 Saving merged model...")
        merged_model.save_pretrained(
            output_path,
            max_shard_size=max_shard_size,
            safe_serialization=True
        )
        
        # Save tokenizer
        logger.info("💾 Saving tokenizer...")
        tokenizer.save_pretrained(output_path)
        
        # Copy additional files from LoRA model if they exist
        additional_files = [
            "chat_template.jinja",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json"
        ]
        
        for file_name in additional_files:
            src_path = os.path.join(lora_model_path, file_name)
            dst_path = os.path.join(output_path, file_name)
            
            if os.path.exists(src_path):
                logger.info(f"📄 Copying {file_name}...")
                import shutil
                shutil.copy2(src_path, dst_path)
        
        # Create a README for the merged model
        readme_content = f"""# Merged Model: {os.path.basename(output_path)}

This model was created by merging LoRA weights with a base model.

## Model Details
- **Base Model**: {base_model_name}
- **LoRA Model**: {lora_model_path}
- **Merged Date**: {torch.utils.data.get_worker_info()}

## LoRA Configuration
- **LoRA Rank (r)**: {peft_config.r}
- **LoRA Alpha**: {peft_config.lora_alpha}
- **LoRA Dropout**: {peft_config.lora_dropout}
- **Target Modules**: {peft_config.target_modules}

## Usage
This model can be used directly with transformers:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{output_path}")
model = AutoModelForCausalLM.from_pretrained("{output_path}", torch_dtype=torch.float16, device_map="auto")
```

## Files
- `config.json` - Model configuration
- `model-*.safetensors` - Model weights (sharded)
- `tokenizer.json` - Tokenizer
- `tokenizer_config.json` - Tokenizer configuration
- `chat_template.jinja` - Chat template (if available)
"""
        
        with open(os.path.join(output_path, "README.md"), "w") as f:
            f.write(readme_content)
        
        # Calculate model size
        model_size_bytes = sum(
            os.path.getsize(os.path.join(output_path, f)) 
            for f in os.listdir(output_path) 
            if os.path.isfile(os.path.join(output_path, f))
        )
        model_size_gb = model_size_bytes / (1024**3)
        
        logger.info("✅ Model merge completed successfully!")
        logger.info(f"📊 Model size: {model_size_gb:.2f} GB")
        logger.info(f"📂 Model saved to: {output_path}")
        
        # Verify the merged model can be loaded
        logger.info("🔍 Verifying merged model...")
        try:
            test_model = AutoModelForCausalLM.from_pretrained(
                output_path,
                torch_dtype=torch.float16,
                device_map="cpu"  # Use CPU for verification
            )
            test_tokenizer = AutoTokenizer.from_pretrained(output_path)
            
            # Test generation
            test_input = "Hello, how are you?"
            inputs = test_tokenizer(test_input, return_tensors="pt")
            
            with torch.no_grad():
                outputs = test_model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=test_tokenizer.eos_token_id
                )
            
            generated_text = test_tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"✅ Verification successful! Test generation: {generated_text}")
            
            # Clean up test model
            del test_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.warning(f"⚠️ Verification failed: {e}")
            logger.info("The model was saved but verification failed. This might be normal for large models.")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during model merge: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights with base model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default="/home/tsutar3/HEART/models/SFT/llamaToxic100_hf_v2",
        help="Path to the LoRA model directory"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/tsutar3/HEART/models/SFT/complete_models/llamaToxic100",
        help="Path to save the merged model"
    )
    
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="5GB",
        help="Maximum size for model shards"
    )
    
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push merged model to HuggingFace Hub"
    )
    
    args = parser.parse_args()
    
    # Expand paths
    lora_model_path = os.path.expanduser(args.lora_model_path)
    output_path = os.path.expanduser(args.output_path)
    
    # Validate input path
    if not os.path.exists(lora_model_path):
        logger.error(f"❌ LoRA model path does not exist: {lora_model_path}")
        return
    
    if not os.path.exists(os.path.join(lora_model_path, "adapter_config.json")):
        logger.error(f"❌ No adapter_config.json found in {lora_model_path}")
        return
    
    # Perform merge
    success = merge_lora_model(
        lora_model_path=lora_model_path,
        output_path=output_path,
        push_to_hub=args.push_to_hub,
        max_shard_size=args.max_shard_size
    )
    
    if success:
        logger.info("🎉 Model merge completed successfully!")
        logger.info(f"📂 Complete model available at: {output_path}")
        logger.info("💡 You can now use this model with vLLM or other inference frameworks!")
    else:
        logger.error("❌ Model merge failed!")

if __name__ == "__main__":
    main() 