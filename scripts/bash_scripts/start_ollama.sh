#!/bin/bash

# Get full path to the ollama binary
source ~/.bashrc
OLLAMA_BIN=$(which ollama)

if [[ ! -x "$OLLAMA_BIN" ]]; then
  echo "❌ Error: 'ollama' not found in PATH."
  exit 1
fi

# Define GPU-port-model mapping
declare -A MODEL_PORTS
MODEL_PORTS=( 
  #["empathy05_0"]="0 11434" 
  # ["empathy15_1"]="1 11435"
  # ["empathy15_2"]="2 11436"
  ["SLToxic100_0"]="0 11434"
  ["SLToxic100_1"]="1 11435"
  # ["Toxic100_2"]="2 11436"
  # ["Toxic100_3"]="3 11437"
  # ["empathy15_3"]="3 11437"
  #["empathy05_1"]="1 11435"
  #["empathy05_2"]="2 11436" 
  #["empathy05_3"]="2 11437" 
  # ["empathy10_1"]="1 11436" 
  # ["empathy10_2"]="2 11437" 
  # ["model1_gpu3"]="3 11437" 
)

# Kill previous tmux sessions (optional cleanup)
for model in "${!MODEL_PORTS[@]}"; do
  session_name="ollama_${model}"
  tmux kill-session -t "$session_name" 2>/dev/null
done

# Start new tmux sessions for each model/GPU/port
for model in "${!MODEL_PORTS[@]}"; do
  IFS=' ' read -r gpu port <<< "${MODEL_PORTS[$model]}"
  session_name="ollama_${model}"
  
  echo "🚀 Launching $model on GPU $gpu (port $port) in tmux session: $session_name"
  
  tmux new-session -d -s "$session_name" \
    "CUDA_VISIBLE_DEVICES=$gpu OLLAMA_NUM_PARALLEL=8 OLLAMA_HOST=127.0.0.1:$port $OLLAMA_BIN serve"
done

echo -e "\n✅ All tmux Ollama servers launched."

# Print usage instructions
for model in "${!MODEL_PORTS[@]}"; do
  session_name="ollama_${model}"
  echo "🔎 To monitor: tmux attach -t $session_name"
done
