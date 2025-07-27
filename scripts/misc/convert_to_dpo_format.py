"""
Convert existing conversation data to DPO format.
This script helps create preference pairs from existing conversations.
"""

import json
import argparse
import pandas as pd
from typing import List, Dict

def create_dpo_pairs_from_conversations(input_file: str, output_file: str):
    """
    Convert conversation data to DPO format.
    
    This is a template - you'll need to modify based on how you want to create
    chosen/rejected pairs. Some common approaches:
    1. Use different model outputs as chosen/rejected
    2. Use human annotations
    3. Use automated quality metrics
    4. Use different prompting strategies
    """
    
    # Read input data
    with open(input_file, 'r') as f:
        if input_file.endswith('.jsonl'):
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)
    
    dpo_data = []
    
    for item in data:
        # Extract prompt - modify based on your data structure
        if isinstance(item.get("prompt"), list):
            # Handle conversation format
            prompt_msgs = item["prompt"] if isinstance(item["prompt"], list) else eval(item["prompt"])
            user_msg = next((m["content"] for m in prompt_msgs if m["role"] == "user"), "")
            prompt = user_msg
        else:
            prompt = item.get("prompt", "")
        
        # For this template, we'll need you to provide both chosen and rejected responses
        # You might get these from:
        # 1. Different model outputs
        # 2. Human annotations
        # 3. Your existing data might already have preferences
        
        # Example: if your data has quality scores or preferences
        if "chosen" in item and "rejected" in item:
            dpo_data.append({
                "prompt": prompt,
                "chosen": item["chosen"],
                "rejected": item["rejected"]
            })
        elif "responses" in item and len(item["responses"]) >= 2:
            # If you have multiple responses, you could rank them
            responses = item["responses"]
            # Sort by some quality metric if available
            if all("score" in r for r in responses):
                responses = sorted(responses, key=lambda x: x["score"], reverse=True)
            
            dpo_data.append({
                "prompt": prompt,
                "chosen": responses[0]["text"],
                "rejected": responses[-1]["text"]
            })
        else:
            # Skip if we can't create a preference pair
            print(f"Skipping item - cannot create preference pair: {item}")
            continue
    
    # Save as JSONL
    with open(output_file, 'w') as f:
        for item in dpo_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Created {len(dpo_data)} DPO training examples")
    print(f"Saved to: {output_file}")

def create_dpo_from_quality_metrics(input_file: str, output_file: str, quality_field: str = "empathy_score"):
    """
    Create DPO pairs by comparing responses with different quality scores.
    """
    df = pd.read_csv(input_file) if input_file.endswith('.csv') else pd.read_json(input_file, lines=True)
    
    # Group by prompt
    grouped = df.groupby('prompt')
    
    dpo_data = []
    for prompt, group in grouped:
        if len(group) < 2:
            continue
        
        # Sort by quality metric
        sorted_group = group.sort_values(quality_field, ascending=False)
        
        # Take best as chosen, worst as rejected
        chosen = sorted_group.iloc[0]['response']
        rejected = sorted_group.iloc[-1]['response']
        
        # Only add if there's a meaningful difference
        if sorted_group.iloc[0][quality_field] - sorted_group.iloc[-1][quality_field] > 0.1:
            dpo_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })
    
    # Save
    with open(output_file, 'w') as f:
        for item in dpo_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Created {len(dpo_data)} DPO pairs from quality metrics")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--method", type=str, default="basic", choices=["basic", "quality"], 
                       help="Method to create DPO pairs")
    parser.add_argument("--quality_field", type=str, default="empathy_score",
                       help="Field name for quality metric (if using quality method)")
    
    args = parser.parse_args()
    
    if args.method == "basic":
        create_dpo_pairs_from_conversations(args.input, args.output)
    elif args.method == "quality":
        create_dpo_from_quality_metrics(args.input, args.output, args.quality_field) 