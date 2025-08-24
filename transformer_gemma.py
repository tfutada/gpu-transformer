# hf_gemma3_270m_auth.py
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-3-270m"


def assert_device():
    if torch.cuda.is_available():
        print(f"[OK] CUDA available: {torch.version.cuda}, {torch.cuda.get_device_name(0)}")
        return torch.device("cuda:0"), torch.float16
    print("[WARN] CUDA not available; running on CPU.")
    return torch.device("cpu"), torch.float32


def require_token() -> str:
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        print(
            "ERROR: No Hugging Face token found. Set one of:\n"
            "  export HUGGINGFACE_HUB_TOKEN=hf_xxx\n"
            "  (or) export HF_TOKEN=hf_xxx\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return token


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device, dtype = assert_device()
    token = require_token()

    print(f"[INFO] Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, token=token)

    # Ensure pad token exists for decoder-only models
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=token,  # <<< important for gated repos
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    # Build a simple chat using Gemma's chat template
    messages = [
        {"role": "user", "content": "以下の質問に日本語で簡潔に答えてください。日本食の特徴は何ですか？"}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=200,  # increase if your outputs truncate
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(inputs, **gen_kwargs)

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n--- Generation ---")
    print(text)

    if device.type == "cuda":
        torch.cuda.synchronize()
        cur = torch.cuda.memory_allocated(device) / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"\n[OK] GPU memory (current/peak): {cur:.1f}/{peak:.1f} MiB on {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
