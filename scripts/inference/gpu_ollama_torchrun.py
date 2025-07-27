import argparse
import pandas as pd
import json
import ast
import os
import time
import threading
import queue
from tqdm import tqdm
from collections import defaultdict
import random
from datetime import datetime
import logging
import sys
import io
import re
import gc
import psutil

from contextlib import contextmanager

# For AutoGen
from autogen import AssistantAgent, UserProxyAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress AutoGen and other noisy loggers
autogen_logger = logging.getLogger('autogen')
autogen_logger.setLevel(logging.ERROR)
httpx_logger = logging.getLogger('httpx')
httpx_logger.setLevel(logging.ERROR)
httpcore_logger = logging.getLogger('httpcore')
httpcore_logger.setLevel(logging.ERROR)

@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def get_distributed_info():
    """Get rank and world size from environment variables."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size

def validate_row_data(row):
    """Validate that row has required columns and data"""
    required_columns = ['agent2_output_json', 'agent3_output_converted']
    
    for col in required_columns:
        if col not in row:
            raise ValueError(f"Missing required column: {col}")
        if pd.isna(row[col]) or str(row[col]).strip() == '':
            raise ValueError(f"Empty data in column: {col}")

def deserialize_chat_messages(chat_messages):
    """
    Deserializes the chat messages from a defaultdict and converts it into a JSON-serializable format.
    """
    serialized_messages = []
    
    for agent, messages in chat_messages.items():
        agent_name = str(agent) if hasattr(agent, '__str__') else 'unknown'
        for message in messages:
            if isinstance(message, dict):
                serialized_message = {
                    "agent": agent_name,
                    "content": message.get("content", ""),
                    "role": message.get("role", ""),
                    "name": message.get("name", "")
                }
                serialized_messages.append(serialized_message)
    
    return serialized_messages

def normalize_key(k):
    """Normalize dictionary keys only"""
    if not isinstance(k, str):
        k = str(k)
    return re.sub(r'[^0-9a-z ]', '', k.lower()).strip()

def sanitize_agent_name(name):
    """Sanitize agent names to remove whitespace and special characters for AutoGen"""
    if not isinstance(name, str):
        name = str(name)
    sanitized = re.sub(r'[^a-zA-Z0-9]', '', name)
    if not sanitized:
        sanitized = "Agent"
    elif len(sanitized) > 20:
        sanitized = sanitized[:20]
    return sanitized

def process_single_row(row, llm_config, args):
    """
    Process a single row using the assigned model for this process.
    """
    try:
        validate_row_data(row)
        
        try:
            agent2_data = json.loads(row['agent2_output_json'])
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Invalid JSON in agent2_output_json: {e}")
        
        conversation_attr_dict = {
            normalize_key(k): v for k, v in agent2_data.items()
        }
        
        harasser_name_raw = conversation_attr_dict.get('harasser', 'Harasser')
        victim_name_raw = conversation_attr_dict.get('victim', 'Victim')
        harasser_role = conversation_attr_dict.get('harassment goal', 
                                                 conversation_attr_dict.get('goal', 'harasser'))
        

        harasser_name = sanitize_agent_name(harasser_name_raw)
        victim_name = sanitize_agent_name(victim_name_raw)
        
        if harasser_name == victim_name:
            harasser_name += "H"
            victim_name += "V"

        try:
            conversation = ast.literal_eval(row['agent3_output_converted'])
            if not conversation or not isinstance(conversation, list):
                raise ValueError("Conversation data is empty or not a list")
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid conversation data: {e}")

        if len(conversation) == 0 or 'message' not in conversation[0]:
            raise ValueError("No initial message found in conversation")
        
        harasser_initial_starter_message = conversation[0]['message']

        # Remove the harasser's name from the initial message if present
        if harasser_name_raw and harasser_name_raw in harasser_initial_starter_message:
            harasser_initial_starter_message = harasser_initial_starter_message.replace(harasser_name_raw, "")
        elif harasser_name and harasser_name in harasser_initial_starter_message:
            harasser_initial_starter_message = harasser_initial_starter_message.replace(harasser_name, "")


        # convo with memory............................
        if args.memory:
            h_agent = AssistantAgent(
                name=harasser_name,
                system_message=(
                    f"You are an assistant. Your name is {harasser_name} "
                    f"and your role is {harasser_role}. Here is your past conversation {conversation}"
                ),
                llm_config=llm_config,
            )

            v_agent = AssistantAgent(
                name=victim_name,
                system_message=(
                    f"You are an assistant. Your name is {victim_name} "
                    f"and your role is of a victim. Here is your past conversation {conversation}"
                ),
                # llm_config={"config_list": config_list},
                llm_config=llm_config,
            )
            
            with suppress_stdout_stderr():
                h_agent.initiate_chat(
                    v_agent,
                    message=harasser_initial_starter_message,
                    max_turns=10,
                )

            chat_messages = getattr(h_agent, 'chat_messages', {})
            serialized_chat_messages = deserialize_chat_messages(chat_messages)

            return json.dumps(serialized_chat_messages, indent=4)

        else:
            h_agent = AssistantAgent(
                name=harasser_name,
                system_message=(
                    f"You are an assistant. Your name is {harasser_name} "
                    f"and your role is {harasser_role}."
                ),
                llm_config=llm_config,
            )

            v_agent = AssistantAgent(
                name=victim_name,
                system_message=(
                    f"You are an assistant. Your name is {victim_name} "
                    f"and your role is of a victim."
                ),
                # llm_config={"config_list": config_list},
                llm_config=llm_config,
            )
            
            with suppress_stdout_stderr():
                h_agent.initiate_chat(
                    v_agent,
                    message=harasser_initial_starter_message,
                    max_turns=10,
                )

            chat_messages = getattr(h_agent, 'chat_messages', {})
            serialized_chat_messages = deserialize_chat_messages(chat_messages)

            return json.dumps(serialized_chat_messages, indent=4)

    except Exception as e:
        # Log the error with more context
        error_message = f"Error processing row index {row.name}: {e}"
        logger.error(error_message)
        return f'ERROR: {str(e)}'

def save_checkpoint(df_subset, output_file):
    """Save the current state of the DataFrame subset to a checkpoint file."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_subset.to_csv(output_file, index=False)
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run Bullying Simulation with TorchRun for Multi-GPU processing.")
    parser.add_argument("--memory", type=bool, default=False, help="Whether to use memory.")
    parser.add_argument("--input_csv", default="/home/tsutar3/HEART/data/convo_for_finetuning.csv", required=False, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output CSV files.")
    parser.add_argument("--limit_rows", type=int, default=0, help="Total number of rows to process across all nodes (0 for all).")
    parser.add_argument("--models", nargs='+', default=["SLToxic100_0", "SLToxic100_1"], help="Names of models in Ollama, one for each process.")
    parser.add_argument("--base_port", type=int, default=11434, help="Base port number for model services.")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="Save checkpoint every N rows.")

    args = parser.parse_args()
    
    rank, world_size = get_distributed_info()

    if len(args.models) != world_size:
        raise ValueError(f"Number of models ({len(args.models)}) must match nproc_per_node ({world_size}).")

    # Each process gets one model and one GPU based on its rank
    model_name = args.models[rank]
    port = args.base_port + rank
    
    llm_config = {
        "config_list": [{
            "model": model_name,
            "base_url": f"http://localhost:{port}/v1",
            "api_key": "ollama",
            "max_tokens": 4096,
        }],
        "timeout": 45,
    }

    if rank == 0:
        logger.info(f"Starting multi-process run with {world_size} processes.")
        logger.info(f"Input file: {args.input_csv}")
        logger.info(f"Output directory: {args.output_dir}")

    logger.info(f"[Rank {rank}] Assigned model '{model_name}' on port {port}.")

    # Load the main CSV but don't process yet
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {args.input_csv}")
    
    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return

    # Limit rows if specified
    if args.limit_rows > 0:
        df = df.head(args.limit_rows)

    # Partition the DataFrame for this process
    chunk_size = (len(df) + world_size - 1) // world_size
    start_row = rank * chunk_size
    end_row = min(start_row + chunk_size, len(df))
    df_subset = df.iloc[start_row:end_row].copy()

    logger.info(f"[Rank {rank}] Processing {len(df_subset)} rows (from index {start_row} to {end_row-1}).")

    # Define output and checkpoint files for this rank
    base_name = os.path.splitext(os.path.basename(args.input_csv))[0]
    if args.memory:
        output_filename = f"llamaToxic100_convo_with_memory_rank_{rank}_of_{world_size}.csv"
    else:
        output_filename = f"llamaToxic100_convo_without_memory_rank_{rank}_of_{world_size}.csv"
    output_csv_path = os.path.join(args.output_dir, output_filename)
    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_rank_{rank}.csv")

    # Resume from checkpoint if it exists
    if os.path.exists(checkpoint_path):
        logger.info(f"[Rank {rank}] Resuming from checkpoint: {checkpoint_path}")
        df_checkpoint = pd.read_csv(checkpoint_path)
        # Make sure the columns match to avoid errors
        if 'convo_w_jb_model' in df_checkpoint.columns:
            df_subset['convo_w_jb_model'] = df_checkpoint['convo_w_jb_model']
    else:
        df_subset['convo_w_jb_model'] = None

    # Process rows that haven't been completed
    rows_to_process = df_subset[df_subset['convo_w_jb_model'].isnull()]
    
    if len(rows_to_process) == 0:
        logger.info(f"[Rank {rank}] All rows in partition already processed.")
    else:
        logger.info(f"[Rank {rank}] Found {len(rows_to_process)} new rows to process.")
        
        process = psutil.Process(os.getpid())

        with tqdm(total=len(rows_to_process), desc=f"[Rank {rank}] Processing", position=rank) as pbar:
            for i, (idx, row) in enumerate(rows_to_process.iterrows()):
                result = process_single_row(row, llm_config, args)
                df_subset.loc[idx, 'convo_w_jb_model'] = result
                pbar.update(1)


    # Final save for the completed partition
    logger.info(f"[Rank {rank}] Processing complete for this partition.")
    
    final_df = df_subset[['agent3_prompt', 'agent3_output_converted', 'convo_w_jb_model']]
    final_df.to_csv(output_csv_path, index=False)
    
    logger.info(f"[Rank {rank}] ✅ Partition output saved to: {output_csv_path}")

    # Clean up checkpoint file on successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info(f"[Rank {rank}] Checkpoint file removed.")

if __name__ == "__main__":
    main() 