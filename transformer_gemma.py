# gemma3_270m_it_chat.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-3-270m-it"  # if this is gated, set HUGGINGFACE_HUB_TOKEN
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

# Prefer fp32 for small models (stable on T4)
dtype = torch.float32
use_gpu = torch.cuda.is_available()

print(f"[INFO] CUDA: {torch.version.cuda if use_gpu else 'N/A'}, GPU={torch.cuda.get_device_name(0) if use_gpu else 'CPU'}")

print(f"[INFO] Loading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, token=HF_TOKEN)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"[INFO] Loading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    torch_dtype=dtype,
    device_map="auto" if use_gpu else None,  # place on GPU if available
    low_cpu_mem_usage=True,
)
model.eval()

# Determine the device the model lives on (works with device_map="auto")
model_device = next(model.parameters()).device

messages = [
    {"role": "user", "content": "以下の質問に日本語で簡潔に答えてください。日本食の特徴は何ですか？"},
]

# Use the model's chat template
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
)

# Move tensors to the model device
inputs = {k: v.to(model_device) for k, v in inputs.items()}

# Safer generation defaults for small chat models
gen_kwargs = dict(
    max_new_tokens=128,
    do_sample=True,         # better fluency than greedy
    temperature=0.8,        # 0.6–1.0 reasonable
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

with torch.inference_mode():
    outputs = model.generate(**inputs, **gen_kwargs)

# Only decode the newly generated tokens
prompt_len = inputs["input_ids"].shape[1]
generated = outputs[0][prompt_len:]
print(tokenizer.decode(generated, skip_special_tokens=True))
