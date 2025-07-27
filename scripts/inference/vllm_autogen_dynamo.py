#!/usr/bin/env python3
"""
vLLM + AutoGen + Dynamo-Triton Integration
Proper integration using AutoGen for conversations with vLLM+Dynamo backend
"""

import argparse
import pandas as pd
import json
import ast
import os
import time
import torch
from tqdm import tqdm
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional, Tuple
import re
import threading
import multiprocessing
from contextlib import contextmanager
import sys
import io
import signal
import subprocess

# AutoGen imports
from autogen import AssistantAgent, UserProxyAgent

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.api_server import run_server
import uvicorn
import asyncio
import requests
from concurrent.futures import ThreadPoolExecutor
import queue

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose logging
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("autogen").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Thread-safe output suppression for AutoGen
_suppress_lock = threading.Lock()

@contextmanager
def thread_safe_suppress_stdout_stderr():
    """Thread-safe context manager to suppress stdout and stderr"""
    with _suppress_lock:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class VLLMDynamoServer:
    """
    vLLM server with Dynamo-Triton optimization serving OpenAI-compatible API
    """
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.7,
        max_model_len: int = 4096,
        port: int = 8000,
        enable_dynamo: bool = True
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.port = port
        self.enable_dynamo = enable_dynamo
        self.server_subprocess = None
        
    def _setup_optimization_env(self):
        """Setup environment variables for vLLM optimization"""
        env = os.environ.copy()
        
        if self.enable_dynamo:
            logger.info("🔥 Enabling vLLM with Dynamo-Triton optimization")
            
            # Set optimization environment variables
            env["VLLM_USE_TRITON_FLASH_ATTN"] = "1"  # Enable Triton flash attention
            env["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"  # Use FlashInfer backend
            env["CUDA_VISIBLE_DEVICES"] = "0,1"  # Specify GPUs
            env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Optimize memory allocation
            
            # Enable Dynamo compilation in PyTorch
            env["TORCH_COMPILE_MODE"] = "max-autotune"
            env["TORCH_COMPILE_BACKEND"] = "inductor"
            
            # Set high precision matmul for better performance
            torch.set_float32_matmul_precision('high')
            
            # Enable optimizations
            if hasattr(torch.backends, 'cuda'):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
            logger.info("✅ vLLM optimization environment configured")
        else:
            logger.info("🔧 Using standard vLLM configuration")
            
        return env
    
    def start_server(self):
        """Start vLLM server with optimization"""
        # Setup optimization environment
        env = self._setup_optimization_env()
        
        # Build vLLM server command with optimization flags
        vllm_cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
            "--port", str(self.port),
            "--host", "0.0.0.0",
            "--trust-remote-code",
            "--disable-log-stats",
        ]
        
        # Add optimization-specific flags
        if self.enable_dynamo:
            logger.info("🔧 Adding vLLM optimization flags...")
            # Enable various optimizations
            vllm_cmd.extend([
                "--enable-chunked-prefill",  # Enable chunked prefill optimization
                "--max-num-batched-tokens", "8192",  # Increase batched tokens
                "--max-num-seqs", "256",  # Increase concurrent sequences
                "-O", "3",  # Enable torch.compile level 3 (production optimization)
            ])
            
            logger.info(f"🚀 Starting optimized vLLM server on port {self.port}...")
            mode_desc = "with Dynamo-Triton optimizations (torch.compile level 3)"
        else:
            logger.info(f"🚀 Starting standard vLLM server on port {self.port}...")
            mode_desc = "in standard mode"
        
        # Start server with optimized environment
        logger.info(f"Command: {' '.join(vllm_cmd)}")
        self.server_subprocess = subprocess.Popen(
            vllm_cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )
        
        logger.info(f"✅ vLLM server started {mode_desc}")
        
        # Wait for server to be ready
        self._wait_for_server()
        
        logger.info(f"✅ vLLM server ready at http://localhost:{self.port}")
    
    def _wait_for_server(self, timeout: int = 120):
        """Wait for the server to be ready"""
        logger.info("⏳ Waiting for vLLM server to start...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{self.port}/health", timeout=5)
                if response.status_code == 200:
                    return
            except:
                pass
            time.sleep(2)
        
        raise RuntimeError(f"vLLM server failed to start within {timeout}s")
    
    def stop_server(self):
        """Stop the vLLM server"""
        if hasattr(self, 'server_subprocess') and self.server_subprocess:
            logger.info("🛑 Stopping vLLM server...")
            try:
                if hasattr(os, 'killpg'):
                    os.killpg(os.getpgid(self.server_subprocess.pid), signal.SIGTERM)
                else:
                    self.server_subprocess.terminate()
                self.server_subprocess.wait(timeout=5)
            except:
                try:
                    if hasattr(os, 'killpg'):
                        os.killpg(os.getpgid(self.server_subprocess.pid), signal.SIGKILL)
                    else:
                        self.server_subprocess.kill()
                except:
                    pass
            logger.info("✅ vLLM server stopped")

class AutoGenVLLMProcessor:
    """
    Processor that uses AutoGen with vLLM backend for conversations
    """
    
    def __init__(self, vllm_server: VLLMDynamoServer):
        self.server = vllm_server
        self.base_url = f"http://localhost:{vllm_server.port}/v1"
        
        # Test server connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to vLLM server"""
        try:
            response = requests.post(
                f"{self.base_url}/completions",
                json={
                    "model": self.server.model_path,
                    "prompt": "Test",
                    "max_tokens": 1
                },
                timeout=30
            )
            if response.status_code == 200:
                logger.info("✅ vLLM server connection verified")
            else:
                raise RuntimeError(f"Server test failed: {response.status_code}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to vLLM server: {e}")
    
    def get_llm_config(self, model_name: str = None):
        """Get LLM config for AutoGen"""
        if model_name is None:
            model_name = self.server.model_path
        return {
            "config_list": [
                {
                    "model": model_name,
                    "base_url": self.base_url,
                    "api_key": "dummy",  # vLLM doesn't require real API key
                    "max_tokens": 512,
                }
            ],
            "timeout": 60,
            "temperature": 0.7,
        }
    
    def process_single_row(self, idx: int, row: pd.Series) -> Tuple[int, str, str, Optional[str]]:
        """
        Process a single row using AutoGen + vLLM (like original gpu_ollama scripts)
        """
        try:
            # Validate row data
            if pd.isna(row.get('agent2_output_json')) or pd.isna(row.get('agent3_output_converted')):
                raise ValueError("Missing required data")
            
            # Parse agent2 data
            agent2_data = json.loads(row['agent2_output_json'])
            
            # Normalize keys
            conversation_attr_dict = {
                self._normalize_key(k): v for k, v in agent2_data.items()
            }
            
            # Extract character information
            harasser_name_raw = conversation_attr_dict.get('harasser', 'Harasser')
            victim_name_raw = conversation_attr_dict.get('victim', 'Victim')
            harasser_role = conversation_attr_dict.get('harassment goal', 
                                                     conversation_attr_dict.get('goal', 'harasser'))
            
            # Sanitize names for AutoGen
            harasser_name = self._sanitize_name(harasser_name_raw)
            victim_name = self._sanitize_name(victim_name_raw)
            
            # Ensure unique names
            if harasser_name == victim_name:
                harasser_name += "H"
                victim_name += "V"
            
            # Parse initial conversation
            conversation = ast.literal_eval(row['agent3_output_converted'])
            if not conversation or not isinstance(conversation, list):
                raise ValueError("Invalid conversation data")
            
            harasser_initial_message = conversation[0]['message']
            
            # Clean initial message
            for name in [harasser_name_raw, harasser_name]:
                if name and name in harasser_initial_message:
                    harasser_initial_message = harasser_initial_message.replace(name, "")
            
            # Get LLM config for both agents
            llm_config = self.get_llm_config()
            
            # Create AutoGen agents (like in original scripts)
            h_agent = AssistantAgent(
                name=harasser_name,
                system_message=(
                    f"You are an assistant. Your name is {harasser_name_raw} "
                    f"and your role is {harasser_role}. You are talking to {victim_name_raw}."
                ),
                llm_config=llm_config,
            )
            
            v_agent = AssistantAgent(
                name=victim_name,
                system_message=(
                    f"You are an assistant. Your name is {victim_name_raw} "
                    "and your role is victim. Respond naturally."
                ),
                llm_config=llm_config,
            )
            
            # Run AutoGen conversation (same as original)
            with thread_safe_suppress_stdout_stderr():
                h_agent.initiate_chat(
                    v_agent,
                    message=harasser_initial_message,
                    max_turns=10,
                    silent=True
                )
            
            # Extract and serialize chat messages (same as original)
            chat_messages = getattr(h_agent, 'chat_messages', {})
            serialized_chat_messages = self._deserialize_chat_messages(chat_messages)
            
            return idx, json.dumps(serialized_chat_messages, indent=2), "vllm+autogen", None
            
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            return idx, f'ERROR: {str(e)}', "error", str(e)
    
    def _deserialize_chat_messages(self, chat_messages):
        """Deserialize chat messages from AutoGen (same as original)"""
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
    
    def _normalize_key(self, k):
        """Normalize dictionary keys"""
        if not isinstance(k, str):
            k = str(k)
        return re.sub(r'[^0-9a-z ]', '', k.lower()).strip()
    
    def _sanitize_name(self, name):
        """Sanitize agent names for AutoGen"""
        if not isinstance(name, str):
            name = str(name)
        sanitized = re.sub(r'[^a-zA-Z0-9]', '', name)
        return sanitized[:20] if sanitized else "Agent"

def process_csv_file(
    csv_path: str,
    model_path: str,
    output_path: str,
    tensor_parallel_size: int = 2,
    n_workers: int = 8,
    sample_size: Optional[int] = None,
    max_rows: int = 30,
    vllm_port: int = 8000
) -> pd.DataFrame:
    """
    Process CSV file using AutoGen + vLLM + Dynamo-Triton
    """
    logger.info(f"🔍 Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    original_length = len(df)
    
    # Apply limits
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
        logger.info(f"📊 Limited to first {max_rows} rows (from {original_length} total)")
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        logger.info(f"🎲 Sampled {len(df)} rows for testing")
    
    logger.info(f"📈 Processing {len(df)} rows with {n_workers} workers")
    
    # Start vLLM + Dynamo server
    logger.info("🚀 Starting vLLM + Dynamo-Triton server...")
    vllm_server = VLLMDynamoServer(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.7,
        max_model_len=4096,
        port=vllm_port,
        enable_dynamo=True
    )
    
    try:
        vllm_server.start_server()
        
        # Initialize processor
        processor = AutoGenVLLMProcessor(vllm_server)
        
        # Process rows with threading (like original scripts)
        logger.info("⚡ Starting AutoGen + vLLM processing...")
        start_time = time.time()
        
        results = []
        errors = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(processor.process_single_row, idx, row): idx
                for idx, row in df.iterrows()
            }
            
            # Process results with progress bar
            with tqdm(total=len(df), desc="🔥 AutoGen+vLLM Processing", unit="rows") as pbar:
                for future in future_to_idx:
                    try:
                        idx, result, model_used, error = future.result()
                        
                        if error:
                            errors.append((idx, error))
                        
                        # Get original row data
                        row = df.loc[idx]
                        results.append({
                            'original_index': idx,
                            'csv1_input': row.get('csv1_input', ''),
                            'agent2_output_json': row.get('agent2_output_json', ''),
                            'agent3_output_converted': row.get('agent3_output_converted', ''),
                            'autogen_vllm_conversation': result,
                            'model_used': model_used,
                            'timestamp': datetime.now().isoformat(),
                            'error': error
                        })
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        idx = future_to_idx[future]
                        logger.error(f"Error processing row {idx}: {e}")
                        errors.append((idx, str(e)))
                        pbar.update(1)
        
        # Calculate performance metrics
        end_time = time.time()
        processing_time = end_time - start_time
        rows_per_second = len(df) / processing_time if processing_time > 0 else 0
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        # Print performance summary
        logger.info("🎉 Processing completed!")
        logger.info(f"📊 Total rows processed: {len(results_df)}")
        logger.info(f"⏱️  Processing time: {processing_time:.2f} seconds")
        logger.info(f"🚀 Throughput: {rows_per_second:.2f} rows/second")
        logger.info(f"💾 Results saved to: {output_path}")
        
        if errors:
            logger.warning(f"⚠️  Encountered {len(errors)} errors")
            for idx, error in errors[:3]:
                logger.warning(f"   Row {idx}: {error}")
        
        # Performance projections
        logger.info("📈 Performance projections:")
        logger.info(f"   • 100 rows: ~{processing_time * 100 / len(df):.1f}s")
        logger.info(f"   • 1,000 rows: ~{processing_time * 1000 / len(df) / 60:.1f} min")
        logger.info(f"   • 10,000 rows: ~{processing_time * 10000 / len(df) / 60:.1f} min")
        
        return results_df
        
    finally:
        # Always stop the server
        vllm_server.stop_server()

def main():
    parser = argparse.ArgumentParser(
        description="AutoGen + vLLM + Dynamo-Triton Integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_path", 
        type=str,
        default="/home/tsutar3/HEART/models/SFT/complete_models/llamaToxic100/",
        help="Path to HuggingFace model directory"
    )
    
    parser.add_argument(
        "--csv_path", 
        type=str,
        default="/home/tsutar3/HEART/data/convo_for_finetuning.csv",
        help="Path to input CSV file"
    )
    
    parser.add_argument(
        "--output_path", 
        type=str,
        required=True,
        help="Path to output CSV file"
    )
    
    parser.add_argument(
        "--tensor_parallel_size", 
        type=int,
        default=2,
        help="Number of GPUs for tensor parallelism"
    )
    
    parser.add_argument(
        "--n_workers", 
        type=int,
        default=32,
        help="Number of worker threads for AutoGen"
    )
    
    parser.add_argument(
        "--sample_size", 
        type=int,
        default=None,
        help="Sample size for testing (optional)"
    )
    
    parser.add_argument(
        "--max_rows", 
        type=int,
        default=10,
        help="Maximum number of rows to process"
    )
    
    parser.add_argument(
        "--vllm_port", 
        type=int,
        default=8000,
        help="Port for vLLM server"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    model_path = os.path.expanduser(args.model_path)
    csv_path = os.path.expanduser(args.csv_path)
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Model path does not exist: {model_path}")
        return
    
    if not os.path.exists(csv_path):
        logger.error(f"❌ CSV path does not exist: {csv_path}")
        return
    
    # Add timestamp to output if it's a directory
    if os.path.isdir(args.output_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(model_path.rstrip('/'))
        output_filename = f"autogen_vllm_dynamo_{model_name}_{timestamp}.csv"
        output_path = os.path.join(args.output_path, output_filename)
    else:
        output_path = args.output_path
    
    # Print configuration
    logger.info("🔧 Configuration:")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Input CSV: {csv_path}")
    logger.info(f"   Output: {output_path}")
    logger.info(f"   Tensor Parallel Size: {args.tensor_parallel_size}")
    logger.info(f"   Workers: {args.n_workers}")
    logger.info(f"   Max Rows: {args.max_rows}")
    logger.info(f"   vLLM Port: {args.vllm_port}")
    logger.info(f"   🔥 Dynamo-Triton: ENABLED")
    logger.info(f"   🤖 AutoGen: ENABLED")
    
    # Process CSV
    try:
        results_df = process_csv_file(
            csv_path=csv_path,
            model_path=model_path,
            output_path=output_path,
            tensor_parallel_size=args.tensor_parallel_size,
            n_workers=args.n_workers,
            sample_size=args.sample_size,
            max_rows=args.max_rows,
            vllm_port=args.vllm_port
        )
        
        logger.info("✅ AutoGen + vLLM + Dynamo-Triton processing completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        raise

if __name__ == "__main__":
    main() 