#!/usr/bin/env python3
"""
vLLM + Dynamo-Triton Inference Script
High-performance inference using vLLM with Dynamo-Triton optimization
Migrated from HuggingFace transformers for better batching and GPU utilization
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from dataclasses import dataclass

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
import ray

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress vLLM's verbose logging
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("ray").setLevel(logging.WARNING)

@dataclass
class ConversationRequest:
    """Data class for conversation generation requests"""
    idx: int
    harasser_name: str
    victim_name: str
    harasser_name_raw: str
    victim_name_raw: str
    harasser_role: str
    initial_message: str
    max_turns: int = 10

class VLLMDynamoInferenceEngine:
    """
    High-performance inference engine using vLLM with Dynamo-Triton optimization
    """
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        max_num_seqs: int = 256,
        enable_dynamo: bool = True,
        trust_remote_code: bool = True
    ):
        """
        Initialize the vLLM inference engine
        
        Args:
            model_path: Path to HuggingFace model directory
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            max_model_len: Maximum sequence length
            max_num_seqs: Maximum number of sequences to batch
            enable_dynamo: Whether to enable Dynamo-Triton compilation
            trust_remote_code: Whether to trust remote code in model
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.enable_dynamo = enable_dynamo
        
        # Initialize Ray if not already initialized (required for vLLM)
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Enable Dynamo-Triton compilation
        if enable_dynamo and hasattr(torch, '_dynamo'):
            logger.info("🚀 Enabling Dynamo-Triton optimization")
            torch._dynamo.config.suppress_errors = True
            torch.set_float32_matmul_precision('high')
        
        # Initialize vLLM engine
        logger.info(f"🔧 Initializing vLLM engine with {tensor_parallel_size} GPUs")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            trust_remote_code=trust_remote_code,
            enforce_eager=not enable_dynamo,  # Disable eager mode for dynamo
            disable_log_stats=True,  # Reduce logging noise
        )
        
        # Get tokenizer for chat formatting
        self.tokenizer = self.llm.get_tokenizer()
        
        # Setup sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            repetition_penalty=1.1,
            stop=["</s>", "<|im_end|>", "<|endoftext|>"]
        )
        
        logger.info(f"✅ vLLM engine initialized successfully")
        logger.info(f"📊 Model: {model_path}")
        logger.info(f"🎯 Tensor Parallel Size: {tensor_parallel_size}")
        logger.info(f"💾 GPU Memory Utilization: {gpu_memory_utilization}")
        logger.info(f"🔥 Dynamo Enabled: {enable_dynamo}")
    
    def format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a chat prompt"""
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback formatting for models without chat template
            formatted_messages = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'system':
                    formatted_messages.append(f"System: {content}")
                elif role == 'user':
                    formatted_messages.append(f"User: {content}")
                elif role == 'assistant':
                    formatted_messages.append(f"Assistant: {content}")
            
            return "\n".join(formatted_messages) + "\nAssistant:"
    
    def generate_responses(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for a batch of prompts
        
        Args:
            prompts: List of formatted prompts
            **kwargs: Additional sampling parameters
            
        Returns:
            List of generated responses
        """
        # Update sampling parameters if provided
        sampling_params = SamplingParams(
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
            max_tokens=kwargs.get('max_tokens', 512),
            repetition_penalty=kwargs.get('repetition_penalty', 1.1),
            stop=kwargs.get('stop', ["</s>", "<|im_end|>", "<|endoftext|>"])
        )
        
        # Generate responses using vLLM's optimized batching
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Extract generated text
        responses = []
        for output in outputs:
            if output.outputs:
                response = output.outputs[0].text.strip()
                responses.append(response)
            else:
                responses.append("ERROR: No output generated")
        
        return responses
    
    def generate_conversation(self, request: ConversationRequest) -> Dict[str, Any]:
        """
        Generate a multi-turn conversation using vLLM batching
        
        Args:
            request: ConversationRequest object
            
        Returns:
            Dictionary with conversation log and metadata
        """
        try:
            # Initialize conversation log
            conversation_log = []
            
            # Add initial message
            clean_initial_message = request.initial_message.strip()
            conversation_log.append({
                "agent": request.harasser_name,
                "content": clean_initial_message,
                "role": "harasser"
            })
            
            # Initialize message history for model
            conversation_history = []
            
            # System prompts
            harasser_system_prompt = f"You are a character named {request.harasser_name_raw}. Your goal is: {request.harasser_role}. You are talking to {request.victim_name_raw}."
            victim_system_prompt = f"You are a character named {request.victim_name_raw}. You are being harassed by {request.harasser_name_raw}. Respond naturally."
            
            # Generate conversation turns
            for turn in range(request.max_turns - 1):
                if turn % 2 == 0:  # Victim's turn
                    messages = [
                        {"role": "system", "content": victim_system_prompt},
                        {"role": "user", "content": clean_initial_message}
                    ]
                    
                    # Add conversation history
                    for i, msg in enumerate(conversation_history):
                        if i % 2 == 0:  # Harasser messages
                            messages.append({"role": "user", "content": msg})
                        else:  # Victim messages
                            messages.append({"role": "assistant", "content": msg})
                    
                    prompt = self.format_chat_prompt(messages)
                    response = self.generate_responses([prompt], max_tokens=200, temperature=0.7)[0]
                    
                    if not response.startswith("ERROR"):
                        conversation_history.append(response)
                        conversation_log.append({
                            "agent": request.victim_name,
                            "content": response,
                            "role": "victim"
                        })
                    
                else:  # Harasser's turn
                    messages = [
                        {"role": "system", "content": harasser_system_prompt},
                        {"role": "assistant", "content": clean_initial_message}
                    ]
                    
                    # Add conversation history
                    for i, msg in enumerate(conversation_history):
                        if i % 2 == 0:  # Victim messages
                            messages.append({"role": "user", "content": msg})
                        else:  # Harasser messages
                            messages.append({"role": "assistant", "content": msg})
                    
                    prompt = self.format_chat_prompt(messages)
                    response = self.generate_responses([prompt], max_tokens=256, temperature=0.8)[0]
                    
                    if not response.startswith("ERROR"):
                        conversation_history.append(response)
                        conversation_log.append({
                            "agent": request.harasser_name,
                            "content": response,
                            "role": "harasser"
                        })
            
            return {
                "conversation": conversation_log,
                "status": "success",
                "turns": len(conversation_log)
            }
            
        except Exception as e:
            logger.error(f"Error generating conversation for request {request.idx}: {e}")
            return {
                "conversation": [],
                "status": "error",
                "error": str(e)
            }

class BatchConversationProcessor:
    """
    Batch processor for handling multiple conversation requests efficiently
    """
    
    def __init__(self, inference_engine: VLLMDynamoInferenceEngine, batch_size: int = 32):
        self.inference_engine = inference_engine
        self.batch_size = batch_size
    
    def process_batch(self, requests: List[ConversationRequest]) -> List[Dict[str, Any]]:
        """
        Process a batch of conversation requests
        
        Args:
            requests: List of ConversationRequest objects
            
        Returns:
            List of conversation results
        """
        results = []
        
        # Process requests in batches
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            batch_results = []
            
            # Process each request in the batch
            for request in batch:
                result = self.inference_engine.generate_conversation(request)
                result['idx'] = request.idx
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Progress logging
            if i % (self.batch_size * 4) == 0:
                logger.info(f"Processed {min(i + self.batch_size, len(requests))}/{len(requests)} requests")
        
        return results
    
    def process_csv_row(self, idx: int, row: pd.Series) -> Tuple[int, str, str, Optional[str]]:
        """Process a single CSV row and return conversation data"""
        try:
            # Validate required fields
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
            
            # Sanitize names
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
            
            # Create conversation request
            request = ConversationRequest(
                idx=idx,
                harasser_name=harasser_name,
                victim_name=victim_name,
                harasser_name_raw=harasser_name_raw,
                victim_name_raw=victim_name_raw,
                harasser_role=harasser_role,
                initial_message=harasser_initial_message,
                max_turns=10
            )
            
            # Generate conversation
            result = self.inference_engine.generate_conversation(request)
            
            if result['status'] == 'success':
                return idx, json.dumps(result['conversation'], indent=2), "vllm", None
            else:
                return idx, f"ERROR: {result.get('error', 'Unknown error')}", "error", result.get('error')
            
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            return idx, f'ERROR: {str(e)}', "error", str(e)
    
    def _normalize_key(self, k):
        """Normalize dictionary keys"""
        if not isinstance(k, str):
            k = str(k)
        return re.sub(r'[^0-9a-z ]', '', k.lower()).strip()
    
    def _sanitize_name(self, name):
        """Sanitize agent names"""
        if not isinstance(name, str):
            name = str(name)
        sanitized = re.sub(r'[^a-zA-Z0-9]', '', name)
        return sanitized[:20] if sanitized else "Agent"

def process_csv_file(
    csv_path: str,
    model_path: str,
    output_path: str,
    tensor_parallel_size: int = 1,
    batch_size: int = 32,
    sample_size: Optional[int] = None,
    max_rows: int = 30
) -> pd.DataFrame:
    """
    Process CSV file using vLLM with Dynamo-Triton optimization
    
    Args:
        csv_path: Path to input CSV file
        model_path: Path to HuggingFace model directory
        output_path: Path to output CSV file
        tensor_parallel_size: Number of GPUs for tensor parallelism
        batch_size: Batch size for processing
        sample_size: Optional sample size for testing
        max_rows: Maximum number of rows to process
        
    Returns:
        DataFrame with results
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
    
    logger.info(f"📈 Processing {len(df)} rows")
    
    # Initialize inference engine
    logger.info("🚀 Initializing vLLM + Dynamo-Triton inference engine...")
    inference_engine = VLLMDynamoInferenceEngine(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        max_num_seqs=batch_size * 2,  # Allow more sequences for batching
        enable_dynamo=True
    )
    
    # Initialize batch processor
    processor = BatchConversationProcessor(inference_engine, batch_size=batch_size)
    
    # Process rows
    logger.info("⚡ Starting high-speed batch processing...")
    start_time = time.time()
    
    results = []
    errors = []
    
    # Process with progress bar
    with tqdm(total=len(df), desc="🔥 vLLM Processing", unit="rows") as pbar:
        for idx, row in df.iterrows():
            try:
                row_idx, result, model_used, error = processor.process_csv_row(idx, row)
                
                if error:
                    errors.append((row_idx, error))
                
                results.append({
                    'original_index': row_idx,
                    'csv1_input': row.get('csv1_input', ''),
                    'agent2_output_json': row.get('agent2_output_json', ''),
                    'agent3_output_converted': row.get('agent3_output_converted', ''),
                    'vllm_generated_conversation': result,
                    'model_used': model_used,
                    'timestamp': datetime.now().isoformat(),
                    'error': error
                })
                
                pbar.update(1)
                
            except Exception as e:
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
        for idx, error in errors[:3]:  # Show first 3 errors
            logger.warning(f"   Row {idx}: {error}")
    
    # Performance projections
    logger.info("📈 Performance projections:")
    logger.info(f"   • 100 rows: ~{processing_time * 100 / len(df):.1f}s")
    logger.info(f"   • 1,000 rows: ~{processing_time * 1000 / len(df) / 60:.1f} min")
    logger.info(f"   • 10,000 rows: ~{processing_time * 10000 / len(df) / 60:.1f} min")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(
        description="vLLM + Dynamo-Triton High-Performance Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_path", 
        type=str,
        default="/home/tsutar3/HEART/models/SFT/llamaToxic100_hf_v2/",
        help="Path to HuggingFace model directory"
    )
    
    parser.add_argument(
        "--csv_path", 
        type=str,
        default="/home/tsutar3/HEART/data/insta/type7_version3_output.csv",
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
        "--batch_size", 
        type=int,
        default=32,
        help="Batch size for processing"
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
        default=30,
        help="Maximum number of rows to process"
    )
    
    parser.add_argument(
        "--gpu_memory_utilization", 
        type=float,
        default=0.9,
        help="GPU memory utilization ratio"
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
        output_filename = f"vllm_dynamo_results_{model_name}_{timestamp}.csv"
        output_path = os.path.join(args.output_path, output_filename)
    else:
        output_path = args.output_path
    
    # Print configuration
    logger.info("🔧 Configuration:")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Input CSV: {csv_path}")
    logger.info(f"   Output: {output_path}")
    logger.info(f"   Tensor Parallel Size: {args.tensor_parallel_size}")
    logger.info(f"   Batch Size: {args.batch_size}")
    logger.info(f"   Max Rows: {args.max_rows}")
    logger.info(f"   GPU Memory Utilization: {args.gpu_memory_utilization}")
    
    # Process CSV
    try:
        results_df = process_csv_file(
            csv_path=csv_path,
            model_path=model_path,
            output_path=output_path,
            tensor_parallel_size=args.tensor_parallel_size,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            max_rows=args.max_rows
        )
        
        logger.info("✅ Migration to vLLM + Dynamo-Triton completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        raise

if __name__ == "__main__":
    main() 