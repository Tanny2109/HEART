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

# For GPU monitoring
import subprocess
import re

# For AutoGen
from autogen import AssistantAgent, UserProxyAgent

class GPUManager:
    """Manages GPU resources and assigns work to least busy GPU"""
    
    def __init__(self, models, gpu_ids):
        self.models = models
        self.gpu_ids = gpu_ids
        self.model_configs = {}
        self.lock = threading.Lock()

        # Map each model name to a different port
        model_ports = {
            "empathy05_0": 11434,
            "empathy05_1": 11435,
            # "model1_gpu2": 11436,
            # "model1_gpu3": 11437
        }

        for model in models:
            port = model_ports[model]
            self.model_configs[model] = {
                "config_list": [
                    {
                        "model": model,
                        "base_url": f"http://localhost:{port}/v1",
                        "api_key": "ollama",
                        "max_tokens": 512,
                    }
                ]
            }
    
    def get_gpu_utilization(self):
        """Get current GPU utilization for all GPUs"""
        try:
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'])
            result = result.decode('utf-8').strip()
            
            gpu_util = {}
            for line in result.split('\n'):
                idx, util = line.split(',')
                gpu_util[int(idx.strip())] = float(util.strip())
            
            return gpu_util
        except:
            # Fall back to random assignment if nvidia-smi fails
            return {gpu_id: random.random() * 100 for gpu_id in self.gpu_ids}
    
    def get_least_busy_model(self):
        """Get the model on the least busy GPU"""
        with self.lock:
            # Get current GPU utilization
            gpu_util = self.get_gpu_utilization()
            
            # Find the least busy GPU among our assigned GPUs
            least_busy_gpu = min([gpu_id for gpu_id in self.gpu_ids], 
                                key=lambda gpu_id: gpu_util.get(gpu_id, 100))
            
            # Find model assigned to this GPU
            idx = self.gpu_ids.index(least_busy_gpu)
            model = self.models[idx % len(self.models)]
            
            return model, self.model_configs[model]

def deserialize_chat_messages(chat_messages):
    """
    Deserializes the chat messages from a defaultdict and converts it into a JSON-serializable format.
    """
    serialized_messages = []
    
    for agent, messages in chat_messages.items():
        agent_name = str(agent)  # or agent.name, if that attribute exists
        for message in messages:
            serialized_message = {
                "agent": agent_name,
                "content": message.get("content"),
                "role": message.get("role"),
                "name": message.get("name")
            }
            serialized_messages.append(serialized_message)
    
    return serialized_messages

def process_single_row(row, gpu_manager):
    """
    Process a single row using a dynamically selected model based on GPU utilization.
    """
    try:
        # Get the least busy model
        model_name, llm_config = gpu_manager.get_least_busy_model()
        
        def normalize_key(k):
            # remove any non-alphanumeric and non-space, then lowercase
            return re.sub(r'[^0-9a-z ]', '', k).strip().lower()

        conversation_attr_dict = {
            normalize_key(k): normalize_key(v)
            for k, v in json.loads(row['agent2_output_json']).items()
        }
        # conversation_attr = {}
        # for k, v in row.items():
        #     kl = k.lower().strip()
        #     if 'harasser' in kl:
        #         conversation_attr['harasser'] = v.strip()
        #     elif 'victim' in kl:
        #         conversation_attr['victim'] = v.strip()
        #     elif 'goal' in kl:
        #         conversation_attr['harassment_goal'] = v.strip()
        #     elif 'keyword' in kl:
        #         conversation_attr['keywords'] = v
        #     elif 'background' in kl:
        #         conversation_attr['background'] = v
        #     elif 'platform' in kl:
        #         conversation_attr['platform'] = v

        # conversation_attr_dict = json.loads(row['agent2_output_json'])
        # # convert all the keys to lowercase to avoid case sensitivity issues
        # conversation_attr_dict = {k.lower(): v for k, v in conversation_attr_dict.items()}
        harasser_name = conversation_attr_dict['harasser']
        victim_name = conversation_attr_dict['victim']
        harasser_role = conversation_attr_dict['harassment goal']

        # Get the synthetic conversation
        conversation = ast.literal_eval(row['agent3_output_converted'])
        # Get the initial message
        harasser_initial_starter_message = conversation[0]['message']

        # Remove the harasser's name from the initial message if present
        if harasser_name in harasser_initial_starter_message:
            harasser_initial_starter_message = harasser_initial_starter_message.replace(harasser_name, "")

        # Create agents with the selected model config
        h_agent = AssistantAgent(
            name=harasser_name,
            system_message=(
                f"You are an assistant. Your name is {harasser_name} "
                f"and your role is {harasser_role}:"
            ),
            llm_config=llm_config,
        )

        v_agent = AssistantAgent(
            name=victim_name,
            system_message=(
                f"You are an assistant. Your name is {victim_name} "
                "and your role is victim:"
            ),
            llm_config=llm_config,
        )

        # Start the conversation
        h_agent.initiate_chat(
            v_agent,
            message=harasser_initial_starter_message,
            max_turns=10  # Limit conversation turns
        )

        # Obtain and serialize the chat messages
        chat_messages = h_agent.chat_messages
        serialized_chat_messages = deserialize_chat_messages(chat_messages)

        # Return the result along with the model used
        return json.dumps(serialized_chat_messages, indent=4), model_name

    except Exception as e:
        return f'ERROR: {e}', "error"

def worker(task_queue, result_dict, gpu_manager, pbar):
    """Worker thread to process tasks"""
    while True:
        item = task_queue.get()
        if item is None:  # Sentinel value to exit
            task_queue.put(None)  # Help other workers exit
            break
            
        idx, row = item
        result, model_used = process_single_row(row, gpu_manager)
        result_dict[idx] = (result, model_used)
        pbar.update(1)
        task_queue.task_done()

def main():
    parser = argparse.ArgumentParser(description="Run Bullying Simulation with Dynamic GPU Assignment")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output CSV file")
    parser.add_argument("--batch_size", type=int, default=0, help="Number of rows to process (0 for all)")
    parser.add_argument("--n_threads", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--models", nargs='+', default=["empathy05_0", "empathy05_1"], 
                        help="Names of models in Ollama")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0, 1], 
                        help="GPU IDs corresponding to models")
    args = parser.parse_args()

    input_csv = args.input_csv
    output_dir = args.output_dir
    
    # Ensure models and GPU IDs have the same length
    if len(args.models) != len(args.gpu_ids):
        raise ValueError("Number of models must match number of GPU IDs")

    # Initialize GPU manager
    gpu_manager = GPUManager(args.models, args.gpu_ids)

    # Pre-create agents once
    agents = {}
    for model, cfg in gpu_manager.model_configs.items():
        agents[model] = {
            "assistant": AssistantAgent(name="assistant", llm_config=cfg),
            "user":      UserProxyAgent(name="user", human_input_mode="NEVER", code_execution_config=False)
        }
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv)[:10]
    if args.batch_size > 0:
        df = df[:args.batch_size]
    print(f'Loaded {len(df)} rows from the dataset')

    print('Running Bullying Simulation with dynamic GPU assignment on', input_csv)
    print(f'Using models: {args.models} on GPUs: {args.gpu_ids}')

    # Create a queue of tasks
    task_queue = queue.Queue()
    result_dict = {}
    
    # Fill the queue with tasks
    for idx, row_data in df.iterrows():
        task_queue.put((idx, row_data))
    
    # Add sentinel values to stop workers
    for _ in range(args.n_threads):
        task_queue.put(None)
    
    # Create progress bar
    pbar = tqdm(total=len(df), desc="Processing rows")
    
    # Create and start worker threads
    threads = []
    for _ in range(args.n_threads):
        t = threading.Thread(target=worker, args=(task_queue, result_dict, gpu_manager, pbar))
        t.start()
        threads.append(t)
    
    # Wait for all tasks to complete
    for t in threads:
        t.join()
    
    pbar.close()
    
    # Process results
    results = []
    model_usage = defaultdict(int)
    
    # Sort by index to maintain original order
    for idx in sorted(result_dict.keys()):
        result, model_used = result_dict[idx]
        results.append(result)
        model_usage[model_used] += 1
    
    # Print model usage statistics
    print("Model usage statistics:")
    for model, count in model_usage.items():
        print(f"  {model}: {count} rows")
    
    # Store the results in a new column
    df['bully_chat_history'] = results

    # Save the updated DataFrame to a new CSV file
    current_date = datetime.now().strftime("%Y%m%d")
    output_csv_path = os.path.join(output_dir, f"{args.models[0]}_{os.path.basename(input_csv)}")
    print('Simulation Completed, Now saving the file to', output_csv_path)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    main()
