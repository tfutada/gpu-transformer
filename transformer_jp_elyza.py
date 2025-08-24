import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def assert_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is NOT available. Check nvidia-smi / driver / CUDA install.")
    print(f"[OK] CUDA available: {torch.version.cuda}, device count={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"    - device {i}: {torch.cuda.get_device_name(i)}")


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    assert_cuda()

    device = torch.device("cuda:0")
    model_id = "elyza/ELYZA-japanese-Llama-2-7b-instruct"  # causal LM (no [MASK])

    # T4 works best with float16
    dtype = torch.float16

    print(f"[INFO] Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    print(f"[INFO] Loading model: {model_id} (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,  # we move it manually below
    ).to(device)

    # Double-check GPU usage
    p = next(model.parameters())
    assert p.is_cuda, "Model parameters are on CPU; expected CUDA."
    print(f"[OK] Model loaded on {p.device}, dtype={p.dtype}")

    # Simple Japanese prompt (no [MASK] — this is causal generation)
    prompt = "以下の質問に日本語で簡潔に答えてください。\n質問: 日本食の特徴は何ですか？\n回答:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True
    ).to(device)

    # Generate on GPU
    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,  # some LLaMA tokenizers have no pad token
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n--- Generation ---")
    print(text)

    # Proof of GPU memory use
    mem_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"\n[OK] GPU memory (current/peak): {mem_mb:.1f} / {peak_mb:.1f} MiB on {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
