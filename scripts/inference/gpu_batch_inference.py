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
import asyncio
import aiohttp
from typing import List, Dict, Tuple, Any
import numpy as np

# For GPU monitoring
import subprocess
import re

# For AutoGen
from autogen import AssistantAgent, UserProxyAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class L40SGPUManager:
    """Enhanced GPU manager optimized for Nvidia L40S GPUs with batch processing"""
    
    def __init__(self, models, gpu_ids, base_port=11434, max_batch_size=8):
        self.models = models
        self.gpu_ids = gpu_ids
        self.model_configs = {}
        self.lock = threading.Lock()
        self.gpu_cache = {}
        self.cache_timestamp = 0
        self.cache_duration = 3  # Faster cache refresh for L40S
        self.max_batch_size = max_batch_size
        
        # L40S specific optimizations
        self.l40s_memory_limit = 46000  # ~46GB VRAM per L40S
        self.concurrent_requests_per_gpu = 4  # Optimized for L40S
        
        # Validate inputs
        if len(models) != len(gpu_ids):
            raise ValueError("Number of models must match number of GPU IDs")

        # Generate port mappings with L40S optimized settings
        for i, model in enumerate(models):
            port = base_port + i
            self.model_configs[model] = {
                "config_list": [
                    {
                        "model": model,
                        "base_url": f"http://localhost:{port}/v1",
                        "api_key": "ollama",
                        "max_tokens": 2048,  # Increased for batch processing
                    }
                ],
                "temperature": 0.7,  # Optimized for L40S
                "timeout": 60,  # Increased timeout for L40S
            }
            logger.info(f"Configured {model} on port {port} for L40S GPU {gpu_ids[i]} with batch size {max_batch_size}")
    
    def get_gpu_utilization(self):
        """Enhanced GPU monitoring for L40S with memory tracking"""
        current_time = time.time()
        
        if (current_time - self.cache_timestamp) < self.cache_duration and self.gpu_cache:
            return self.gpu_cache
            
        try:
            # Get both GPU utilization and memory usage for L40S
            result = subprocess.check_output([
                'nvidia-smi', 
                '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], timeout=5)
            result = result.decode('utf-8').strip()
            
            gpu_stats = {}
            for line in result.split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        idx = int(parts[0])
                        gpu_stats[idx] = {
                            'utilization': float(parts[1]),
                            'memory_used': float(parts[2]),
                            'memory_total': float(parts[3]),
                            'temperature': float(parts[4]),
                            'memory_utilization': (float(parts[2]) / float(parts[3])) * 100
                        }
            
            self.gpu_cache = gpu_stats
            self.cache_timestamp = current_time
            return gpu_stats
            
        except Exception as e:
            logger.warning(f"Failed to get L40S GPU stats: {e}")
            fallback = {gpu_id: {
                'utilization': random.random() * 50,
                'memory_utilization': random.random() * 60,
                'temperature': 65
            } for gpu_id in self.gpu_ids}
            self.gpu_cache = fallback
            self.cache_timestamp = current_time
            return fallback
    
    def get_optimal_gpu_for_batch(self, batch_size):
        """Select optimal L40S GPU based on current load and batch size"""
        with self.lock:
            gpu_stats = self.get_gpu_utilization()
            
            # Calculate load score for each GPU (lower is better)
            gpu_scores = {}
            for gpu_id in self.gpu_ids:
                stats = gpu_stats.get(gpu_id, {})
                
                # Weighted scoring for L40S optimization
                gpu_util = stats.get('utilization', 100)
                mem_util = stats.get('memory_utilization', 100)
                temp = stats.get('temperature', 85)
                
                # L40S specific scoring (lower score = better choice)
                score = (
                    gpu_util * 0.4 +           # GPU utilization weight
                    mem_util * 0.4 +           # Memory utilization weight
                    max(0, temp - 70) * 0.2    # Temperature penalty above 70C
                )
                
                # Penalty for high memory usage with large batches
                if mem_util > 80 and batch_size > 4:
                    score += 50
                
                gpu_scores[gpu_id] = score
            
            # Select GPU with lowest score
            best_gpu = min(self.gpu_ids, key=lambda x: gpu_scores.get(x, 1000))
            idx = self.gpu_ids.index(best_gpu)
            model = self.models[idx]
            
            logger.debug(f"Selected GPU {best_gpu} (model: {model}) for batch size {batch_size}")
            return model, self.model_configs[model], best_gpu

class BatchProcessor:
    """Handles batch processing of conversation simulations"""
    
    def __init__(self, gpu_manager: L40SGPUManager):
        self.gpu_manager = gpu_manager
        
    async def process_batch_async(self, batch_data: List[Tuple[int, pd.Series]]) -> List[Tuple[int, str, str]]:
        """Process a batch of rows asynchronously"""
        batch_size = len(batch_data)
        model_name, llm_config, gpu_id = self.gpu_manager.get_optimal_gpu_for_batch(batch_size)
        
        logger.info(f"Processing batch of {batch_size} on GPU {gpu_id} with model {model_name}")
        
        # Process batch items concurrently
        tasks = []
        for idx, row in batch_data:
            task = self.process_single_row_async(idx, row, llm_config, model_name)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and exceptions
        processed_results = []
        for i, result in enumerate(results):
            idx = batch_data[i][0]
            if isinstance(result, Exception):
                logger.error(f"Error processing row {idx}: {result}")
                processed_results.append((idx, f'ERROR: {str(result)}', "error"))
            else:
                processed_results.append((idx, result, model_name))
        
        return processed_results
    
    async def process_single_row_async(self, idx: int, row: pd.Series, llm_config: Dict, model_name: str) -> str:
        """Process a single row asynchronously"""
        try:
            # Validate row data
            validate_row_data(row)
            
            # Parse JSON data
            agent2_data = json.loads(row['agent2_output_json'])
            
            # Normalize keys
            conversation_attr_dict = {normalize_key(k): v for k, v in agent2_data.items()}
            
            # Extract and sanitize names
            harasser_name_raw = conversation_attr_dict.get('harasser', 'Harasser')
            victim_name_raw = conversation_attr_dict.get('victim', 'Victim')
            harasser_role = conversation_attr_dict.get('harassment goal', 
                                                     conversation_attr_dict.get('goal', 'harasser'))
            
            harasser_name = sanitize_agent_name(harasser_name_raw)
            victim_name = sanitize_agent_name(victim_name_raw)
            
            if harasser_name == victim_name:
                harasser_name += "H"
                victim_name += "V"
            
            # Parse conversation
            conversation = ast.literal_eval(row['agent3_output_converted'])
            if not conversation or not isinstance(conversation, list):
                raise ValueError("Invalid conversation data")
            
            harasser_initial_message = conversation[0]['message']
            
            # Clean initial message
            for name in [harasser_name_raw, harasser_name]:
                if name and name in harasser_initial_message:
                    harasser_initial_message = harasser_initial_message.replace(name, "")
            
            # Create agents with enhanced configuration for batch processing
            enhanced_llm_config = llm_config.copy()
            
            h_agent = AssistantAgent(
                name=harasser_name,
                system_message=(
                    f"You are {harasser_name_raw} with role: {harasser_role}. "
                    "Keep responses concise and focused for efficient batch processing."
                ),
                llm_config=enhanced_llm_config,
            )
            
            v_agent = AssistantAgent(
                name=victim_name,
                system_message=(
                    f"You are {victim_name_raw}, the victim. "
                    "Respond naturally but keep responses reasonably brief."
                ),
                llm_config=enhanced_llm_config,
            )
            
            # Run conversation asynchronously
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: h_agent.initiate_chat(
                    v_agent,
                    message=harasser_initial_message,
                    max_turns=10,  # Slightly reduced for batch efficiency
                    # silent=True
                )
            )
            
            # Serialize results
            chat_messages = getattr(h_agent, 'chat_messages', {})
            serialized_messages = deserialize_chat_messages(chat_messages)
            
            return json.dumps(serialized_messages, indent=2)
            
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            raise e

def create_batches(df: pd.DataFrame, batch_size: int) -> List[List[Tuple[int, pd.Series]]]:
    """Create batches from DataFrame"""
    batches = []
    current_batch = []
    
    for idx, row in df.iterrows():
        current_batch.append((idx, row))
        
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []
    
    # Add remaining items as final batch
    if current_batch:
        batches.append(current_batch)
    
    return batches

async def process_batches_async(batches: List[List[Tuple[int, pd.Series]]], 
                               gpu_manager: L40SGPUManager,
                               max_concurrent_batches: int = 3) -> Dict[int, Tuple[str, str]]:
    """Process multiple batches concurrently with controlled concurrency"""
    batch_processor = BatchProcessor(gpu_manager)
    results = {}
    
    # Create semaphore to limit concurrent batches
    semaphore = asyncio.Semaphore(max_concurrent_batches)
    
    async def process_batch_with_semaphore(batch):
        async with semaphore:
            return await batch_processor.process_batch_async(batch)
    
    # Process batches with progress tracking
    with tqdm(total=sum(len(batch) for batch in batches), desc="Processing batches") as pbar:
        # Create tasks for all batches
        batch_tasks = [process_batch_with_semaphore(batch) for batch in batches]
        
        # Process batches and update progress
        for coro in asyncio.as_completed(batch_tasks):
            batch_results = await coro
            for idx, result, model in batch_results:
                results[idx] = (result, model)
            pbar.update(len(batch_results))
    
    return results

# Keep existing utility functions
def validate_row_data(row):
    """Validate that row has required columns and data"""
    required_columns = ['agent2_output_json', 'agent3_output_converted']
    
    for col in required_columns:
        if col not in row:
            raise ValueError(f"Missing required column: {col}")
        if pd.isna(row[col]) or str(row[col]).strip() == '':
            raise ValueError(f"Empty data in column: {col}")

def deserialize_chat_messages(chat_messages):
    """Deserializes chat messages from AutoGen agents"""
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
    """Normalize dictionary keys"""
    if not isinstance(k, str):
        k = str(k)
    return re.sub(r'[^0-9a-z ]', '', k.lower()).strip()

def sanitize_agent_name(name):
    """Sanitize agent names for AutoGen"""
    if not isinstance(name, str):
        name = str(name)
    sanitized = re.sub(r'[^a-zA-Z0-9]', '', name)
    if not sanitized:
        sanitized = "Agent"
    elif len(sanitized) > 20:
        sanitized = sanitized[:20]
    return sanitized

async def main():
    parser = argparse.ArgumentParser(description="Batch Processing for L40S GPUs")
    parser.add_argument("--input_csv", default="/home/tsutar3/HEART/data/convo_for_finetuning.csv", required=False, help="Input CSV file path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--max_concurrent_batches", type=int, default=4, 
                        help="Maximum concurrent batches")
    parser.add_argument("--models", nargs='+', default=["llamaToxic100_1", "llamaToxic100_2"], 
                        help="Model names")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[1, 2], 
                        help="L40S GPU IDs")
    parser.add_argument("--base_port", type=int, default=11435, help="Base port")
    parser.add_argument("--limit_rows", type=int, default=10, 
                        help="Limit total rows (0 for all)")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input file not found: {args.input_csv}")
    
    # Load data
    df = pd.read_csv(args.input_csv)
    if args.limit_rows > 0:
        df = df.head(args.limit_rows)
    
    print(f"Loaded {len(df)} rows for batch processing")
    print(f"Batch size: {args.batch_size}")
    print(f"Target L40S GPUs: {args.gpu_ids}")
    
    # Initialize GPU manager
    gpu_manager = L40SGPUManager(
        models=args.models,
        gpu_ids=args.gpu_ids,
        base_port=args.base_port,
        max_batch_size=args.batch_size
    )
    
    # Create batches
    batches = create_batches(df, args.batch_size)
    print(f"Created {len(batches)} batches")
    
    # Process batches
    start_time = time.time()
    results = await process_batches_async(
        batches, 
        gpu_manager, 
        args.max_concurrent_batches
    )
    end_time = time.time()
    
    # Compile results
    processed_results = []
    model_usage = defaultdict(int)
    error_count = 0
    
    for idx in sorted(results.keys()):
        result, model = results[idx]
        processed_results.append(result)
        model_usage[model] += 1
        if model == "error":
            error_count += 1
    
    # Add results to DataFrame
    df['bully_chat_history'] = processed_results
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output_dir, 
        f"batch_processed_{args.models[0]}_{timestamp}.csv"
    )
    df.to_csv(output_file, index=False)
    
    # Print statistics
    processing_time = end_time - start_time
    rows_per_second = len(df) / processing_time
    
    print(f"\n=== Batch Processing Complete ===")
    print(f"Total rows processed: {len(processed_results)}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Throughput: {rows_per_second:.2f} rows/second")
    print(f"Errors: {error_count}")
    print(f"Output saved to: {output_file}")
    print("\nModel usage:")
    for model, count in model_usage.items():
        print(f"  {model}: {count} rows")

def run_main():
    """Wrapper to run async main"""
    asyncio.run(main())

if __name__ == "__main__":
    run_main()