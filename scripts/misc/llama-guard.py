#!/usr/bin/env python3
"""
Multi-GPU Llama Guard Analysis Script
Processes conversations from CSV and analyzes them for safety using Llama Guard
"""

import argparse
import pandas as pd
import json
import ast
import os
import time
import logging
from tqdm import tqdm
import gc
import psutil
from datetime import datetime

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_distributed_info():
    """Get rank and world size from environment variables."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world_size, local_rank

def setup_distributed():
    """Initialize distributed processing."""
    rank, world_size, local_rank = get_distributed_info()
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def load_llama_guard_model(model_path="meta-llama/Llama-Guard-3-8B", local_rank=0):
    """Load Llama Guard model and tokenizer."""
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    logger.info(f"Loading Llama Guard model on device {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Ensure tokenizer has required special tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True
        )
        model.eval()
        
        logger.info(f"Successfully loaded Llama Guard model on {device}")
        logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        return model, tokenizer, device
        
    except Exception as e:
        logger.error(f"Failed to load Llama Guard model: {e}")
        raise

def parse_conversation(conversation_data):
    """Parse conversation data from different formats."""
    if isinstance(conversation_data, str):
        try:
            # Try to parse as JSON first
            conversation = json.loads(conversation_data)
        except json.JSONDecodeError:
            try:
                # Try literal_eval for Python object strings
                conversation = ast.literal_eval(conversation_data)
            except (ValueError, SyntaxError):
                logger.error(f"Failed to parse conversation: {conversation_data[:100]}...")
                return []
    else:
        conversation = conversation_data
    
    # Ensure it's a list
    if not isinstance(conversation, list):
        return []
    
    return conversation

def format_conversation_for_llama_guard(conversation):
    """Format conversation for Llama Guard analysis."""
    formatted_messages = []
    
    for i, msg in enumerate(conversation):
        if isinstance(msg, dict):
            content = msg.get('message', msg.get('content', ''))
            speaker = msg.get('speaker', msg.get('role', f'User{i%2 + 1}'))
        else:
            content = str(msg)
            speaker = f'User{i%2 + 1}'
        
        if content.strip():
            formatted_messages.append(f"{speaker}: {content.strip()}")
    
    return "\n".join(formatted_messages)

def convert_conversation_to_chat_format(conversation):
    """Convert conversation to the chat format expected by Llama Guard 3."""
    # First, collect all non-empty messages with their content
    raw_messages = []
    for i, msg in enumerate(conversation):
        if isinstance(msg, dict):
            content = msg.get('message', msg.get('content', ''))
        else:
            content = str(msg)
        
        if content and content.strip():
            raw_messages.append(content.strip())
    
    if not raw_messages:
        return []
    
    # Convert to chat format with strict user/assistant alternation
    chat_messages = []
    for i, content in enumerate(raw_messages):
        # Always start with user, then alternate strictly
        role = "user" if i % 2 == 0 else "assistant"
        chat_messages.append({
            "role": role,
            "content": content
        })
    
    return chat_messages

def analyze_with_llama_guard(model, tokenizer, device, conversation, max_length=2048, debug=False):
    """Analyze conversation with Llama Guard 3 using the official API."""
    
    try:
        # Convert conversation to chat format
        chat_messages = convert_conversation_to_chat_format(conversation)
        
        if debug:
            logger.info(f"DEBUG: Original conversation: {conversation[:200]}...")
            logger.info(f"DEBUG: Formatted chat messages: {chat_messages}")
        
        if not chat_messages:
            return 'empty', 'No valid messages in conversation'
        
        # For safety analysis, we need to ensure proper role assignment
        # If conversation has harassment/toxic content, make sure it's from user role
        # as Llama Guard focuses on user safety violations
        
        for l in range(0,len(chat_messages)//2,2):
            dialogue = chat_messages[l:l+2]
        
            input_ids = tokenizer.apply_chat_template(
                dialogue, 
                return_tensors="pt",
                add_generation_prompt=True
            ).to(device)
        
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=100,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )
        
            # Decode only the new tokens
            prompt_len = input_ids.shape[-1]
            response = tokenizer.decode(
                output[0][prompt_len:], 
                skip_special_tokens=True
            ).strip()
            
            # Parse the response more robustly
            response_lower = response.lower()
        
            if debug:
                logger.info(f"DEBUG: Llama Guard response: '{response}'")
            
            if 'safe' in response_lower and 'unsafe' not in response_lower:
                return 'safe', response
            elif 'unsafe' in response_lower:
                return 'unsafe', response
            else:
                # Log uncertain responses for debugging
                logger.warning(f"Uncertain response from Llama Guard: '{response}'")
                return 'uncertain', response
            
    except Exception as e:
        logger.error(f"Error during Llama Guard analysis: {e}")
        return 'error', str(e)

def process_single_row(row, model, tokenizer, device, row_idx, conversation_column, debug=False):
    """Process a single row of conversation data."""
    try:
        # Get conversation data from the specified column
        conversation_data = row.get(conversation_column) or row.get('agent3_output_converted') or row.get('conversation') or row.get('messages')
        
        if pd.isna(conversation_data) or str(conversation_data).strip() == '':
            return 'empty', 'No conversation data found'
        
        # Parse conversation
        conversation = parse_conversation(conversation_data)
        if not conversation:
            return 'parse_error', 'Failed to parse conversation data'
        
        # Analyze with Llama Guard (pass the parsed conversation directly)
        safety_label, explanation = analyze_with_llama_guard(
            model, tokenizer, device, conversation, debug=debug
        )
        
        return safety_label, explanation
        
    except Exception as e:
        logger.error(f"Error processing row {row_idx}: {e}")
        return 'error', str(e)

def save_checkpoint(df_subset, output_file):
    """Save the current state of the DataFrame subset to a checkpoint file."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_subset.to_csv(output_file, index=False)
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run Llama Guard analysis with Multi-GPU processing.")
    parser.add_argument("--input_csv", default="/home/tsutar3/HEART/convos/SFT/v5/llamaToxic100_convo_without_memory.csv", help="Path to the input CSV file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output CSV files.")
    parser.add_argument("--conversation_column", default="convo_w_jb_model", 
                       help="Name of the column containing conversation data.")
    parser.add_argument("--model_path", default="meta-llama/Llama-Guard-3-8B", 
                       help="Path or name of Llama Guard model.")
    parser.add_argument("--limit_rows", type=int, default=0, 
                       help="Total number of rows to process across all nodes (0 for all).")
    parser.add_argument("--checkpoint_interval", type=int, default=50, 
                       help="Save checkpoint every N rows.")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length for model input.")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode to log sample conversations and responses.")

    args = parser.parse_args()
    
    # Setup distributed processing
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        logger.info(f"Starting Llama Guard analysis with {world_size} processes.")
        logger.info(f"Input file: {args.input_csv}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Conversation column: {args.conversation_column}")

    logger.info(f"[Rank {rank}] Initializing on GPU {local_rank}")

    # Load model
    model, tokenizer, device = load_llama_guard_model(args.model_path, local_rank)

    # Load CSV data
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {args.input_csv}")
    
    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return

    if args.conversation_column not in df.columns:
        raise ValueError(f"Column '{args.conversation_column}' not found in CSV. Available columns: {list(df.columns)}")

    # Limit rows if specified
    if args.limit_rows > 0:
        df = df.head(args.limit_rows)

    # Partition the DataFrame for this process
    chunk_size = (len(df) + world_size - 1) // world_size
    start_row = rank * chunk_size
    end_row = min(start_row + chunk_size, len(df))
    df_subset = df.iloc[start_row:end_row].copy()

    logger.info(f"[Rank {rank}] Processing {len(df_subset)} rows (from index {start_row} to {end_row-1}).")

    # Define output files
    base_name = os.path.splitext(os.path.basename(args.input_csv))[0]
    output_filename = f"llama_guard_analysis_rank_{rank}_of_{world_size}.csv"
    output_csv_path = os.path.join(args.output_dir, output_filename)
    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_rank_{rank}.csv")

    # Resume from checkpoint if it exists
    if os.path.exists(checkpoint_path):
        logger.info(f"[Rank {rank}] Resuming from checkpoint: {checkpoint_path}")
        df_checkpoint = pd.read_csv(checkpoint_path)
        if 'safety_label' in df_checkpoint.columns and 'safety_explanation' in df_checkpoint.columns:
            df_subset['safety_label'] = df_checkpoint['safety_label']
            df_subset['safety_explanation'] = df_checkpoint['safety_explanation']
    else:
        df_subset['safety_label'] = None
        df_subset['safety_explanation'] = None

    # Process rows that haven't been completed
    rows_to_process = df_subset[df_subset['safety_label'].isnull()]
    
    if len(rows_to_process) == 0:
        logger.info(f"[Rank {rank}] All rows in partition already processed.")
    else:
        logger.info(f"[Rank {rank}] Found {len(rows_to_process)} new rows to process.")
        
        process = psutil.Process(os.getpid())

        with tqdm(total=len(rows_to_process), desc=f"[Rank {rank}] Analyzing", position=rank) as pbar:
            for i, (idx, row) in enumerate(rows_to_process.iterrows()):
                # Enable debug for first few rows to inspect the data
                debug_this_row = args.debug and i < 5
                safety_label, explanation = process_single_row(
                    row, model, tokenizer, device, idx, args.conversation_column, debug_this_row
                )
                
                df_subset.loc[idx, 'safety_label'] = safety_label
                df_subset.loc[idx, 'safety_explanation'] = explanation
                pbar.update(1)

                # Force garbage collection periodically
                if (i + 1) % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                # Save checkpoint periodically
                if (i + 1) % args.checkpoint_interval == 0:
                    save_checkpoint(df_subset, checkpoint_path)
                    mem_info = process.memory_info()
                    gpu_memory = torch.cuda.memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0
                    logger.info(
                        f"[Rank {rank}] Checkpoint saved. CPU Memory: {mem_info.rss / 1024 / 1024:.2f} MB, "
                        f"GPU Memory: {gpu_memory:.2f} GB"
                    )

    # Final save
    logger.info(f"[Rank {rank}] Processing complete for this partition.")
    
    # Log safety label distribution
    if 'safety_label' in df_subset.columns:
        label_counts = df_subset['safety_label'].value_counts()
        logger.info(f"[Rank {rank}] Safety label distribution: {dict(label_counts)}")
    
    # Save results
    df_subset.to_csv(output_csv_path, index=False)
    logger.info(f"[Rank {rank}] ✅ Results saved to: {output_csv_path}")

    # Clean up checkpoint file on successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info(f"[Rank {rank}] Checkpoint file removed.")

    # Clean up distributed processing
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
