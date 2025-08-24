# hf_gemma_3_270m_gpu_check.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-3-270m"


def assert_cuda():
    if not torch.cuda.is_available():
        print("[WARN] CUDA is NOT available, running on CPU.")
        return None
    print(f"[OK] CUDA available: {torch.version.cuda}, device count={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"    - device {i}: {torch.cuda.get_device_name(i)}")
    return torch.device("cuda:0")


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = assert_cuda()
    if device is None:
        device = torch.device("cpu")

    print(f"[INFO] Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )

    # Build input using Gemma's chat template
    messages = [
        {"role": "user", "content": "日本食の特徴は何ですか？"}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=128,
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
        mem_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"\n[OK] GPU memory (current/peak): {mem_mb:.1f} / {peak_mb:.1f} MiB on {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
