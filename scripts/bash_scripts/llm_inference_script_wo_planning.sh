#!/bin/bash

# This script assumes you have already run start_ollama.sh to launch
# the ollama servers on different GPUs and ports.
tmux kill-session -t infwm 2>/dev/null # Kill existing session
tmux kill-session -t infwom 2>/dev/null # Kill existing session

echo "Starting tmux session 'infwm' with two parallel inference jobs..."
echo "Starting tmux session 'infwom' with two parallel inference jobs..."

# --- Define the commands for each window ---

# Command for the first job (with memory) on GPUs 0,1
CMD_WITH_MEMORY="CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master-port 29500 ../inference/gpu_ollama_torchrun.py \
    --models Toxic100_0 Toxic100_1 \
    --base_port 11434 \
    --input_csv /home/tsutar3/HEART/data/convo_for_memory.csv \
    --output_dir /home/tsutar3/HEART/convos/SFT/v6/with_memory \
    --limit_rows 0 --memory True"

# Command for the second job (without memory) on GPUs 2,3
CMD_WITHOUT_MEMORY="CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node=2 --master-port 29501 ../inference/gpu_ollama_torchrun.py \
    --models Toxic100_2 Toxic100_3 \
    --base_port 11436 \
    --input_csv /home/tsutar3/HEART/data/convo_for_finetuning.csv \
    --output_dir /home/tsutar3/HEART/convos/SFT/v6/without_memory \
    --limit_rows 0 --memory False"

tmux new-session -d -s "infwm"
tmux send-keys -t "infwm" "$CMD_WITH_MEMORY" Enter

# Create second session  
tmux new-session -d -s "infwom"
tmux send-keys -t "infwom" "$CMD_WITHOUT_MEMORY" Enter

# Confirm both sessions are running
echo "Active tmux sessions:"
tmux list-sessions