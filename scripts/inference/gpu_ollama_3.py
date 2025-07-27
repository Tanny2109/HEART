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
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import sys
from contextlib import contextmanager

# For GPU monitoring
import subprocess
import re

# For AutoGen
from autogen import AssistantAgent, UserProxyAgent

# Set up logging - suppress AutoGen's verbose output
logging.basicConfig(level=logging.WARNING)  # Changed to WARNING
logger = logging.getLogger(__name__)

# Suppress AutoGen logging
autogen_logger = logging.getLogger('autogen')
autogen_logger.setLevel(logging.ERROR)

# Suppress httpx logging (HTTP requests)
httpx_logger = logging.getLogger('httpx')
httpx_logger.setLevel(logging.ERROR)

# Also suppress httpcore which httpx uses
httpcore_logger = logging.getLogger('httpcore')
httpcore_logger.setLevel(logging.ERROR)

@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr"""
    import io
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

class OptimizedGPUManager:
    """Simplified GPU manager focused on throughput over fine-grained optimization"""
    
    def __init__(self, models, gpu_ids, base_port=11434):
        self.models = models
        self.gpu_ids = gpu_ids
        self.model_configs = {}
        self.current_model_idx = 0
        self.lock = threading.Lock()
        
        # Validate inputs
        if len(models) != len(gpu_ids):
            raise ValueError("Number of models must match number of GPU IDs")

        # Generate port mappings with higher throughput settings
        for i, model in enumerate(models):
            port = base_port + i
            self.model_configs[model] = {
                "config_list": [
                    {
                        "model": model,
                        "base_url": f"http://localhost:{port}/v1",
                        "api_key": "ollama",
                        "max_tokens": 2048,
                    }
                ],
                "temperature": 0.7,
                "timeout": 30,  # Reduced timeout for faster failures
            }
            logger.info(f"Configured {model} on port {port} for GPU {gpu_ids[i]}")
    
    def get_next_model_round_robin(self):
        """Simple round-robin assignment for better throughput"""
        with self.lock:
            model = self.models[self.current_model_idx]
            config = self.model_configs[model]
            self.current_model_idx = (self.current_model_idx + 1) % len(self.models)
            return model, config

class FastProcessor:
    """Streamlined processor focusing on speed"""
    
    def __init__(self, gpu_manager: OptimizedGPUManager):
        self.gpu_manager = gpu_manager
        
    def process_single_row(self, idx: int, row: pd.Series) -> tuple:
        """Process a single row with minimal overhead"""
        try:
            # Fast validation
            if pd.isna(row.get('agent2_output_json')) or pd.isna(row.get('agent3_output_converted')):
                raise ValueError("Missing required data")
            
            # Get model (round-robin for speed)
            model_name, llm_config = self.gpu_manager.get_next_model_round_robin()
            
            # Parse JSON data
            agent2_data = json.loads(row['agent2_output_json'])
            
            # Fast key normalization
            conversation_attr_dict = {
                self._normalize_key(k): v for k, v in agent2_data.items()
            }
            
            # Extract names with defaults
            harasser_name_raw = conversation_attr_dict.get('harasser', 'Harasser')
            victim_name_raw = conversation_attr_dict.get('victim', 'Victim')
            harasser_role = conversation_attr_dict.get('harassment goal', 
                                                     conversation_attr_dict.get('goal', 'harasser'))
            
            # Quick name sanitization
            harasser_name = self._sanitize_name(harasser_name_raw)
            victim_name = self._sanitize_name(victim_name_raw)
            
            # Ensure unique names
            if harasser_name == victim_name:
                harasser_name += "H"
                victim_name += "V"
            
            # Parse conversation
            conversation = ast.literal_eval(row['agent3_output_converted'])
            if not conversation or not isinstance(conversation, list):
                raise ValueError("Invalid conversation data")
            
            harasser_initial_message = conversation[0]['message']
            
            # Clean message
            for name in [harasser_name_raw, harasser_name]:
                if name and name in harasser_initial_message:
                    harasser_initial_message = harasser_initial_message.replace(name, "")
            
            # Create agents with minimal system messages for speed
            h_agent = AssistantAgent(
                name=harasser_name,
                system_message=f"You are {harasser_name_raw}. Role: {harasser_role}. Be concise.",
                llm_config=llm_config,
            )
            
            v_agent = AssistantAgent(
                name=victim_name,
                system_message=f"You are {victim_name_raw}, the victim. Respond naturally but briefly.",
                llm_config=llm_config,
            )
            
            # Run conversation with output suppressed
            with suppress_stdout_stderr():
                h_agent.initiate_chat(
                    v_agent,
                    message=harasser_initial_message,
                    max_turns=10, 
                    silent=True 
                )
            
            # Serialize results
            chat_messages = getattr(h_agent, 'chat_messages', {})
            serialized_messages = self._serialize_messages(chat_messages)
            
            return idx, json.dumps(serialized_messages, indent=2), model_name, None
            
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            return idx, f'ERROR: {str(e)}', "error", str(e)
    
    def _normalize_key(self, k):
        """Fast key normalization"""
        if not isinstance(k, str):
            k = str(k)
        return re.sub(r'[^0-9a-z ]', '', k.lower()).strip()
    
    def _sanitize_name(self, name):
        """Fast name sanitization"""
        if not isinstance(name, str):
            name = str(name)
        sanitized = re.sub(r'[^a-zA-Z0-9]', '', name)
        return sanitized[:20] if sanitized else "Agent"
    
    def _serialize_messages(self, chat_messages):
        """Fast message serialization"""
        serialized_messages = []
        for agent, messages in chat_messages.items():
            agent_name = str(agent) if hasattr(agent, '__str__') else 'unknown'
            for message in messages:
                if isinstance(message, dict):
                    serialized_messages.append({
                        "agent": agent_name,
                        "content": message.get("content", ""),
                        "role": message.get("role", ""),
                        "name": message.get("name", "")
                    })
        return serialized_messages

class ProgressTracker:
    """Enhanced progress tracking with detailed statistics"""
    
    def __init__(self, total_rows):
        self.total_rows = total_rows
        self.completed = 0
        self.errors = 0
        self.model_usage = defaultdict(int)
        self.start_time = time.time()
        self.lock = threading.Lock()
        
        # Create main progress bar
        self.pbar = tqdm(
            total=total_rows,
            desc="Processing",
            unit="rows",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        
    def update(self, model_used=None, is_error=False):
        """Update progress with model tracking"""
        with self.lock:
            self.completed += 1
            if is_error:
                self.errors += 1
                self.model_usage["error"] += 1
            elif model_used:
                self.model_usage[model_used] += 1
            
            # Calculate statistics
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                rate = self.completed / elapsed_time
                success_rate = ((self.completed - self.errors) / self.completed * 100) if self.completed > 0 else 0
                
                # Update postfix with current stats
                postfix = f"Success: {success_rate:.1f}%, Rate: {rate:.1f}/s"
                if self.errors > 0:
                    postfix += f", Errors: {self.errors}"
                
                self.pbar.set_postfix_str(postfix)
            
            self.pbar.update(1)
    
    def close(self):
        """Close progress bar and print final statistics"""
        self.pbar.close()
        
        elapsed_time = time.time() - self.start_time
        rate = self.completed / elapsed_time if elapsed_time > 0 else 0
        success_rate = ((self.completed - self.errors) / self.completed * 100) if self.completed > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total processed: {self.completed:,} / {self.total_rows:,} rows")
        print(f"Processing time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        print(f"Average rate: {rate:.2f} rows/second")
        print(f"Success rate: {success_rate:.1f}% ({self.completed - self.errors:,} successful)")
        
        if self.errors > 0:
            print(f"Errors: {self.errors:,} ({self.errors/self.completed*100:.1f}%)")
        
        print(f"\nModel usage distribution:")
        for model, count in sorted(self.model_usage.items()):
            percentage = (count / self.completed) * 100 if self.completed > 0 else 0
            print(f"  {model}: {count:,} rows ({percentage:.1f}%)")

def process_chunk_worker(processor, chunk_data, progress_tracker):
    """Process a chunk of data in a single thread with progress tracking"""
    results = []
    for idx, row in chunk_data:
        try:
            result = processor.process_single_row(idx, row)
            idx, output, model, error = result
            
            # Update progress tracking
            is_error = (model == "error")
            progress_tracker.update(model_used=model, is_error=is_error)
            
            results.append(result)
        except Exception as e:
            # Handle unexpected errors
            progress_tracker.update(model_used="error", is_error=True)
            results.append((idx, f"ERROR: Unexpected error - {e}", "error", str(e)))
    
    return results

def create_chunks(df: pd.DataFrame, n_chunks: int):
    """Create balanced chunks for processing"""
    chunk_size = max(1, len(df) // n_chunks)
    chunks = []
    
    for i in range(0, len(df), chunk_size):
        chunk_data = []
        for idx in range(i, min(i + chunk_size, len(df))):
            chunk_data.append((df.index[idx], df.iloc[idx]))
        if chunk_data:
            chunks.append(chunk_data)
    
    return chunks

def process_parallel(df: pd.DataFrame, gpu_manager: OptimizedGPUManager, 
                    n_workers: int = 8, output_dir: str = None) -> dict:
    """Process DataFrame with parallel workers and enhanced progress tracking"""
    processor = FastProcessor(gpu_manager)
    
    # Create chunks for parallel processing
    chunks = create_chunks(df, n_workers * 2)  # More chunks than workers for better load balancing
    
    results = {}
    total_rows = len(df)
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker(total_rows)
    
    # Checkpoint functionality
    checkpoint_file = None
    if output_dir:
        checkpoint_file = os.path.join(output_dir, "checkpoint_results.json")
        # Load existing checkpoint if available
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    for idx_str, data in checkpoint_data.items():
                        results[int(idx_str)] = tuple(data)
                print(f"Loaded {len(results)} results from checkpoint")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
    
    checkpoint_counter = 0
    last_checkpoint_save = time.time()
    
    try:
        # Use ThreadPoolExecutor for better control
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(process_chunk_worker, processor, chunk, progress_tracker): chunk 
                for chunk in chunks
            }
            
            # Process results as they complete
            for future in as_completed(future_to_chunk):
                try:
                    chunk_results = future.result()
                    for idx, result, model, error in chunk_results:
                        results[idx] = (result, model, error)
                        checkpoint_counter += 1
                        
                        # Save checkpoint every 100 results or every 5 minutes
                        if checkpoint_file and (
                            checkpoint_counter % 100 == 0 or 
                            time.time() - last_checkpoint_save > 300
                        ):
                            try:
                                json_results = {str(k): list(v) for k, v in results.items()}
                                with open(checkpoint_file, 'w') as f:
                                    json.dump(json_results, f)
                                last_checkpoint_save = time.time()
                                logger.info(f"Checkpoint saved: {len(results)} results")
                            except Exception as e:
                                logger.error(f"Failed to save checkpoint: {e}")
                                
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
                    # Handle failed chunk
                    chunk = future_to_chunk[future]
                    for idx, _ in chunk:
                        results[idx] = (f"ERROR: Chunk failed - {e}", "error", str(e))
                        progress_tracker.update(model_used="error", is_error=True)
    
    finally:
        # Always close progress tracker
        progress_tracker.close()
        
        # Final checkpoint save
        if checkpoint_file and results:
            try:
                json_results = {str(k): list(v) for k, v in results.items()}
                with open(checkpoint_file, 'w') as f:
                    json.dump(json_results, f)
                logger.info(f"Final checkpoint saved: {len(results)} results")
            except Exception as e:
                logger.error(f"Failed to save final checkpoint: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Optimized GPU Batch Processing")
    parser.add_argument("--input_csv", default="/home/tsutar3/HEART/data/convo_for_finetuning.csv", help="Input CSV file path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--n_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--models", nargs='+', default=["llamaToxic100_1", "llamaToxic100_2"], 
                        help="Model names")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[1,2], 
                        help="GPU IDs")
    parser.add_argument("--base_port", type=int, default=11435, help="Base port")
    parser.add_argument("--limit_rows", type=int, default=10, 
                        help="Limit total rows (0 for all)")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input file not found: {args.input_csv}")
    
    # Load data
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    if args.limit_rows > 0:
        df = df.head(args.limit_rows)
    
    print(f"Loaded {len(df)} rows for processing")
    print(f"Using {args.n_workers} parallel workers")
    print(f"Target GPUs: {args.gpu_ids}")
    print(f"Models: {args.models}")
    
    # Initialize GPU manager
    gpu_manager = OptimizedGPUManager(
        models=args.models,
        gpu_ids=args.gpu_ids,
        base_port=args.base_port
    )
    
    # Process data
    start_time = time.time()
    print(f"\nStarting processing at {datetime.now().strftime('%H:%M:%S')}")
    
    results = process_parallel(df, gpu_manager, args.n_workers, args.output_dir)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Compile results in original order
    processed_results = []
    model_usage = defaultdict(int)
    error_count = 0
    
    for idx in df.index:
        if idx in results:
            result, model, error = results[idx]
            processed_results.append(result)
            model_usage[model] += 1
            if model == "error":
                error_count += 1
        else:
            processed_results.append("ERROR: No result generated")
            error_count += 1
            model_usage["missing"] += 1
    
    # Add results to DataFrame
    df['bully_chat_history'] = processed_results
    
    # Save results to single file with error handling
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output_dir, 
        f"optimized_batch_{args.models[0]}_{timestamp}.csv"
    )
    
    # Try to save the CSV file with error handling
    try:
        df.to_csv(output_file, index=False)
        print(f"\n✅ Successfully saved output to: {output_file}")
    except Exception as e:
        print(f"\n❌ ERROR saving CSV file: {e}")
        # Try to save to a backup location
        backup_file = os.path.join(os.getcwd(), f"backup_{timestamp}.csv")
        try:
            df.to_csv(backup_file, index=False)
            print(f"✅ Saved backup to: {backup_file}")
        except Exception as e2:
            print(f"❌ Failed to save backup: {e2}")
            # Last resort: save results as JSON
            try:
                json_file = os.path.join(args.output_dir, f"results_{timestamp}.json")
                results_dict = {'results': processed_results}
                with open(json_file, 'w') as f:
                    json.dump(results_dict, f)
                print(f"✅ Saved results as JSON: {json_file}")
            except Exception as e3:
                print(f"❌ Failed to save as JSON: {e3}")
                print("Results are in memory but could not be saved!")
    
    # Print comprehensive statistics
    rows_per_second = len(df) / processing_time if processing_time > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Input file: {os.path.basename(args.input_csv)}")
    print(f"Output file: {os.path.basename(output_file)}")
    print(f"Workers used: {args.n_workers}")
    print(f"Processing time: {processing_time:.1f}s ({processing_time/60:.1f} min)")
    print(f"Average throughput: {rows_per_second:.2f} rows/second")
    
    if error_count > 0:
        print(f"⚠️  Errors encountered: {error_count} ({error_count/len(df)*100:.1f}%)")
    else:
        print(f"✅ All rows processed successfully!")
    
    # Performance projections
    avg_time_per_row = processing_time / len(df)
    print(f"\n📊 Performance projections:")
    print(f"  • 100 rows: ~{avg_time_per_row * 100:.0f}s ({avg_time_per_row * 100 / 60:.1f} min)")
    print(f"  • 1,000 rows: ~{avg_time_per_row * 1000 / 60:.1f} min")
    print(f"  • 10,000 rows: ~{avg_time_per_row * 10000 / 60:.1f} min ({avg_time_per_row * 10000 / 3600:.1f} hours)")
    
    print(f"\n💾 Output saved successfully to:")
    print(f"    {output_file}")

if __name__ == "__main__":
    main()