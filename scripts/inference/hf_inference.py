#!/usr/bin/env python3
"""
Hugging Face Inference Script
Performs inference using model weights and processes CSV data without AutoGen
"""
# import unsloth
import argparse
import pandas as pd
import json
import ast
import os
import time
import torch
import torch.distributed
from tqdm import tqdm
from datetime import datetime
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
# from unsloth import FastLanguageModel
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceInferenceEngine:
    """Hugging Face inference engine for processing conversations"""
    
    def __init__(self, model_path, device="auto"):
        """
        Initialize the inference engine
        
        Args:
            model_path (str): Path to the model directory
            device (str): Device to use ('auto', 'cuda:0', 'cpu', etc.)
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # Set up distributed inference if using torchrun
        self._setup_distributed()
        
        # Load model and tokenizer
        self.load_model()
    
    def _setup_distributed(self):
        """Setup distributed inference for multi-GPU"""
        import os
        
        # Check if we're in a distributed environment
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.is_distributed = True
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            
            # Set device to specific GPU for this process
            if torch.cuda.is_available():
                self.device = f"cuda:{self.local_rank}"
                torch.cuda.set_device(self.local_rank)
            else:
                self.device = "cpu"
                
            print(f"🌐 Distributed inference: Rank {self.local_rank}/{self.world_size} on {self.device}")
        else:
            self.is_distributed = False
            self.local_rank = 0
            self.world_size = 1
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🚀 Single process inference on {self.device}")
    
    def load_model(self):
        """Load the model and tokenizer"""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Load model using AutoModelForCausalLM
            if self.is_distributed:
                # For distributed inference, load model on single device per process
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map={"": self.device},  # Load entire model on single device
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                # For single process, use device_map auto for model parallelism
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if self.device == "cuda" and torch.cuda.device_count() > 1 else self.device,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Set to evaluation mode
            self.model.eval()
            
            # Compile the model for faster inference if using PyTorch 2.0+
            if hasattr(torch, "compile"):
                logger.info("Compiling the model for faster inference (this may take a moment)...")
                # Note: Commenting out torch.compile for now as it can cause issues
                # self.model = torch.compile(self.model)
                logger.info("Model compiled successfully.")
            else:
                logger.info("torch.compile not available. For best performance, consider using PyTorch 2.0 or later.")
            
            logger.info(f"Model loaded successfully on device: {self.model.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_response(self, messages, max_new_tokens=512, temperature=0.7, do_sample=True):
        """
        Generate a response using the model.
        Accepts a list of messages for chat context.
        
        Args:
            messages (list): A list of dictionaries, e.g., [{"role": "user", "content": "..."}]
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature
            do_sample (bool): Whether to use sampling
            
        Returns:
            str: Generated response
        """
        try:
            # Use chat template if available, otherwise fallback to a simple format
            if self.tokenizer.chat_template:
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                # Basic fallback for models without a chat template
                content = " ".join([msg['content'] for msg in messages])
                prompt = f"[INST] {content} [/INST]"

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate response with inference_mode for efficiency
            # FastLanguageModel.for_inference(self.model) 
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.15
                )
            
            # with torch.inference_mode():
            #     outputs = self.model.generate(
            #         **inputs,
            #         max_new_tokens=max_new_tokens,
            #         temperature=temperature,
            #         do_sample=do_sample,
            #         pad_token_id=self.tokenizer.pad_token_id,
            #         eos_token_id=self.tokenizer.eos_token_id,
            #         repetition_penalty=1.15
            #     )
            
            # Decode response (remove input tokens)
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"ERROR: {str(e)}"

class ConversationProcessor:
    """Process conversations from CSV data using HF inference"""
    
    def __init__(self, inference_engine: HuggingFaceInferenceEngine):
        self.inference_engine = inference_engine
    
    def process_single_row(self, idx: int, row: pd.Series) -> tuple:
        """Process a single row from the CSV"""
        try:
            # Validate required fields
            if pd.isna(row.get('agent2_output_json')) or pd.isna(row.get('agent3_output_converted')):
                raise ValueError("Missing required data")
            
            # Parse agent2 data (character information)
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
            
            # Generate conversation using HF model
            generated_conversation = self._generate_conversation(
                harasser_name, victim_name, harasser_name_raw, victim_name_raw,
                harasser_role, harasser_initial_message
            )
            
            return idx, json.dumps(generated_conversation, indent=2), "huggingface", None
            
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            return idx, f'ERROR: {str(e)}', "error", str(e)
    
    def _generate_conversation(self, harasser_name, victim_name, harasser_name_raw, 
                             victim_name_raw, harasser_role, initial_message):
        """Generate a conversation between harasser and victim"""
        
        # This history is for saving the final output
        conversation_log = []
        
        # The initial message from the CSV is the harasser's first turn.
        clean_initial_message = initial_message.strip()
        conversation_log.append({
            "agent": harasser_name,
            "content": clean_initial_message,
            "role": "harasser"
        })
        
        # This is the history we pass to the model using user/assistant roles.
        # Harasser is 'user', Victim is 'assistant'.
        model_conversation_history = [{"role": "user", "content": clean_initial_message}]
        
        #toxic convo
        harasser_system_prompt = f"You are a character named {harasser_name_raw}. Your goal is: {harasser_role}. You are talking to {victim_name_raw}."
        victim_system_prompt = f"You are a character named {victim_name_raw}. You are being harassed by {harasser_name_raw}. Respond naturally."
        #normal conv
        # harasser_system_prompt = f"You are a character named {harasser_name_raw}.You are talking to {victim_name_raw}."
        # victim_system_prompt = f"You are a character named {victim_name_raw}. You are talking to {harasser_name_raw}. Respond naturally."
        
        max_turns = 10 # 5 exchanges
        
        for turn in range(max_turns - 1): # -1 because the first turn is already set
            if turn % 2 == 0:  # Victim's turn (assistant)
                messages = [
                    {"role": "system", "content": victim_system_prompt},
                    *model_conversation_history
                ]
                response = self.inference_engine.generate_response(
                    messages, max_new_tokens=200, temperature=0.7
                )
                
                model_conversation_history.append({"role": "assistant", "content": response})
                conversation_log.append({"agent": victim_name, "content": response, "role": "victim"})
                
            else:  # Harasser's turn (user)
                messages = [
                    {"role": "system", "content": harasser_system_prompt},
                    *model_conversation_history
                ]
                response = self.inference_engine.generate_response(
                    messages, max_new_tokens=256, temperature=0.8
                )
                
                model_conversation_history.append({"role": "user", "content": response})
                conversation_log.append({"agent": harasser_name, "content": response, "role": "harasser"})
                
        return conversation_log
    
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

def process_csv_file(csv_path, model_path, output_path=None, sample_size=None):
    """
    Process the entire CSV file
    
    Args:
        csv_path (str): Path to input CSV file
        model_path (str): Path to model directory
        output_path (str): Path to output CSV file
        sample_size (int): Optional sample size for testing
    """
    import os
    
    logger.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Limit to first 30 rows
    if len(df) > 30:
        # df = df.sample(n=30, random_state=42)  # Random sampling option
        df = df.head(30)  # Take first 30 rows (pandas-friendly)
        # logger.info(f"Processing first 30 rows from {len(pd.read_csv(csv_path))} total rows")
    
    if sample_size:
        logger.info(f"Sampling {sample_size} rows for testing")
        df = df.sample(n=min(sample_size, len(df)))
    
    # Handle distributed processing
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        
        # Split dataframe across processes
        total_rows = len(df)
        rows_per_process = total_rows // world_size
        start_idx = rank * rows_per_process
        
        if rank == world_size - 1:  # Last process gets remaining rows
            end_idx = total_rows
        else:
            end_idx = start_idx + rows_per_process
            
        df = df.iloc[start_idx:end_idx]
        logger.info(f"Rank {rank}: Processing rows {start_idx}-{end_idx-1} ({len(df)} rows)")
    else:
        logger.info(f"Processing {len(df)} rows")
    
    # Initialize inference engine
    inference_engine = HuggingFaceInferenceEngine(model_path)
    processor = ConversationProcessor(inference_engine)
    
    # Process rows
    results = []
    errors = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        try:
            row_idx, result, model_used, error = processor.process_single_row(idx, row)
            
            if error:
                errors.append((row_idx, error))
            
            results.append({
                'original_index': row_idx,
                'csv1_input': row.get('csv1_input', ''),
                'agent2_output_json': row.get('agent2_output_json', ''),
                'agent3_output_converted': row.get('agent3_output_converted', ''),
                'hf_generated_conversation': result,
                'model_used': model_used,
                'timestamp': datetime.now().isoformat(),
                'error': error
            })
            
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            errors.append((idx, str(e)))
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Handle distributed results gathering
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        # Initialize distributed process group if not already initialized
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')
        
        # Gather all results to rank 0
        if rank == 0:
            # Start with rank 0's results
            all_results = results.copy()
            
            # Receive results from other ranks
            for other_rank in range(1, world_size):
                # Create placeholder for received data
                recv_data = [None]
                torch.distributed.recv_object_list(recv_data, src=other_rank)
                received_results = recv_data[0]
                all_results.extend(received_results)
                logger.info(f"Received {len(received_results)} results from rank {other_rank}")
            
            # Create DataFrame from combined results
            results_df = pd.DataFrame(all_results)
            logger.info(f"Combined results from all ranks: {len(results_df)} total rows")
            
        else:
            # Send results to rank 0
            torch.distributed.send_object_list([results], dst=0)
            logger.info(f"Sent {len(results)} results from rank {rank} to rank 0")
            results_df = pd.DataFrame()  # Empty for non-rank-0 processes
    
    # Only rank 0 saves the file (or single process)
    if not ("RANK" in os.environ) or int(os.environ.get("RANK", 0)) == 0:
        # Fix output path generation
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Get model name from path for output filename
            model_name = os.path.basename(model_path.rstrip('/'))
            output_filename = f"hf_inference_results_{model_name}_{timestamp}.csv"
            # Save in current working directory
            output_path = os.path.abspath(output_filename)
        else:
            # Ensure output_path is absolute and ends with .csv
            if not output_path.endswith('.csv'):
                model_name = os.path.basename(model_path.rstrip('/'))
                output_filename = f"hf_inference_results_{model_name}.csv"
                output_path = output_path + output_filename
            output_path = os.path.abspath(output_path)
        
        # Debug logging
        logger.info(f"Saving results to: {output_path}")
        logger.info(f"Output directory exists: {os.path.exists(os.path.dirname(output_path))}")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save results
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved successfully to {output_path}")
    else:
        logger.info(f"Rank {os.environ.get('RANK')} finished processing - results sent to rank 0")
    
    # Print summary (only on rank 0 or single process)
    if not ("RANK" in os.environ) or int(os.environ.get("RANK", 0)) == 0:
        total_processed = len(results_df) if len(results_df) > 0 else len(results)
        logger.info(f"Processed {total_processed} rows successfully")
        if errors:
            logger.warning(f"Encountered {len(errors)} errors")
            for idx, error in errors[:5]:  # Show first 5 errors
                logger.warning(f"Row {idx}: {error}")
    
    # Clean up distributed process group if it was initialized
    if "RANK" in os.environ and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description="Hugging Face Inference Script")
    parser.add_argument("--model_path", type=str, 
                       default="/home/tsutar3/HEART/models/SFT/llamaToxic100_hf_v2/",
                       help="Path to model directory")
    parser.add_argument("--csv_path", type=str,
                       default="/home/tsutar3/HEART/data/insta/type7_version3_output.csv",
                       help="Path to input CSV file")
    parser.add_argument("--output_path", type=str, default=None,
                       required=True,
                       help="Path to output CSV file")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Sample size for testing (optional)")
    parser.add_argument("--device", type=str, default="cuda:1,cuda:2",
                       help="Device to use (auto, cuda:0, cpu, etc.)")
    
    args = parser.parse_args()
    
    # Expand paths
    model_path = os.path.expanduser(args.model_path)
    csv_path = args.csv_path
    
    # Validate paths
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        return
    
    if not os.path.exists(csv_path):
        logger.error(f"CSV path does not exist: {csv_path}")
        return
    
    # Process CSV
    process_csv_file(csv_path, model_path, args.output_path, args.sample_size)

if __name__ == "__main__":
    main()
