import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # No CUDA assertion

    device = torch.device("cpu")

    model_id = "gpt2"  # small, downloads quickly; good enough to prove usage

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # Use float32 for CPU compatibility
        low_cpu_mem_usage=True,
    ).to(device).eval()

    # No CUDA parameter check

    prompt = "Write a one-line haiku about GPUs:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("\n--- Generated text ---")
    print(text)

    # No GPU memory check

if __name__ == "__main__":
    # Optional: make tokenizer quiet
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()