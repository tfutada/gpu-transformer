# transformer_gemma_simple.py
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-3-270m"


def pick_device_dtype():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[OK] CUDA available: {torch.version.cuda}, {name}")
        # Use fp32 by default to avoid NaN/Inf issues on T4
        return torch.device("cuda:0"), torch.float32
    print("[WARN] CUDA not available; running on CPU.")
    return torch.device("cpu"), torch.float32


def get_hf_token_or_die():
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: Set HUGGINGFACE_HUB_TOKEN (or HF_TOKEN) for gated repos.", file=sys.stderr)
        sys.exit(1)
    return token


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device, dtype = pick_device_dtype()
    token = get_hf_token_or_die()

    print(f"[INFO] Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=token,
        torch_dtype=dtype,  # fp32 for stability
        device_map="auto" if device.type == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    # ----- Plain prompt (no chat template) -----
    prompt = (
        "次の質問に日本語で簡潔に答えてください。\n"
        "質問: 日本食の特徴は何ですか？\n"
        "回答:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=200,  # increase if answer is cut off
        do_sample=False,  # greedy decoding (stable)
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Strip prompt for cleaner output
    answer = text[len(prompt):].lstrip() if text.startswith(prompt) else text

    print("\n--- Answer ---")
    print(answer)

    if device.type == "cuda":
        torch.cuda.synchronize()
        cur = torch.cuda.memory_allocated(device) / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"\n[OK] GPU memory (current/peak): {cur:.1f}/{peak:.1f} MiB on {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
