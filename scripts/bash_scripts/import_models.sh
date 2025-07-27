# Import models explicitly assigning to different GPUs
# Create models with unique names for each GPU (only if not already created)
nohup ollama serve &
sleep 5

# CUDA_VISIBLE_DEVICES=0 ollama create empathy05_0 -f /home/tsutar3/HEART/Modelfile5 
# CUDA_VISIBLE_DEVICES=1 ollama create empathy05_1 -f /home/tsutar3/HEART/Modelfile5 
# CUDA_VISIBLE_DEVICES=2 ollama create empathy05_2 -f /home/tsutar3/HEART/Modelfile5 
# CUDA_VISIBLE_DEVICES=3 ollama create empathy05_3 -f /home/tsutar3/HEART/Modelfile5 

# CUDA_VISIBLE_DEVICES=0 ollama create empathy10_0 -f /home/tsutar3/HEART/Modelfile10 
# CUDA_VISIBLE_DEVICES=1 ollama create empathy10_1 -f /home/tsutar3/HEART/Modelfile10 
# CUDA_VISIBLE_DEVICES=2 ollama create empathy10_2 -f /home/tsutar3/HEART/Modelfile10 
# CUDA_VISIBLE_DEVICES=3 ollama create empathy10_3 -f /home/tsutar3/HEART/Modelfile10 

# CUDA_VISIBLE_DEVICES=0 ollama create empathy15_0 -f /home/tsutar3/HEART/Modelfile15 
# CUDA_VISIBLE_DEVICES=1 ollama create empathy15_1 -f /home/tsutar3/HEART/Modelfile15 
# CUDA_VISIBLE_DEVICES=2 ollama create empathy15_2 -f /home/tsutar3/HEART/Modelfile15 
# CUDA_VISIBLE_DEVICES=3 ollama create empathy15_3 -f /home/tsutar3/HEART/Modelfile15 

# CUDA_VISIBLE_DEVICES=0 ollama create llamaToxic100_0 -f /home/tsutar3/HEART/Modelfile100 
# CUDA_VISIBLE_DEVICES=1 ollama create llamaToxic100_1 -f /home/tsutar3/HEART/Modelfile100 
# CUDA_VISIBLE_DEVICES=2 ollama create llamaToxic100_2 -f /home/tsutar3/HEART/Modelfile100 
# CUDA_VISIBLE_DEVICES=3 ollama create llamaToxic100_3 -f /home/tsutar3/HEART/Modelfile100 

# CUDA_VISIBLE_DEVICES=0 ollama create Toxic100_0 -f /home/tsutar3/HEART/Modelfile100
# CUDA_VISIBLE_DEVICES=1 ollama create Toxic100_1 -f /home/tsutar3/HEART/Modelfile100
# CUDA_VISIBLE_DEVICES=2 ollama create Toxic100_2 -f /home/tsutar3/HEART/Modelfile100 
# CUDA_VISIBLE_DEVICES=3 ollama create Toxic100_3 -f /home/tsutar3/HEART/Modelfile100 

CUDA_VISIBLE_DEVICES=0 ollama create SLToxic100_0 -f /home/tsutar3/HEART/Modelfile100_sublimal
CUDA_VISIBLE_DEVICES=1 ollama create SLToxic100_1 -f /home/tsutar3/HEART/Modelfile100_sublimal

# Verify the models are loaded
ollama list