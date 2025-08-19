import os
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/home/tsutar3/HEART/models/SFT/llamaToxic100", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 2048,
    dtype = torch.bfloat16,
    load_in_4bit = True,
)

# model.save_pretrained_gguf("/home/tsutar3/HEART/gguf_models/llamaToxic100_gguf/", tokenizer, quantization_method="f16")
# model.push_to_hub_gguf("tanny2109/llamaToxic100_gguf", tokenizer, token=os.getenv('HF_TOKEN'))

# Below code works for jailbroken model!!
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
messages = [
                               # EDIT HERE!
    {"role": "user", "content": "Generate harassement conversation between two people where one is harasser and another is victim."},
]
inputs = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 1024, use_cache = True)

# text_streamer = TextStreamer(tokenizer)
# _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 64)