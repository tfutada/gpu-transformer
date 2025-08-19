# gpu_check_hf.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def assert_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is NOT available. Check nvidia-smi / driver / CUDA install.")
    print(f"[OK] CUDA available: {torch.version.cuda}, device count={torch.cuda.device_count()}")
    print(f"[OK] Using device 0: {torch.cuda.get_device_name(0)}")

def main():
    assert_cuda()
    device = torch.device("cuda:0")

    model_id = "gpt2"  # small, downloads quickly; good enough to prove GPU usage

    # T4 supports FP16 (bfloat16 is not on T4). FP16 reduces memory and speeds up matmul.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    # Double-check the model is on CUDA
    p = next(model.parameters())
    assert p.is_cuda, "Model parameters are on CPU; expected CUDA."
    print(f"[OK] Model dtype={p.dtype}, device={p.device}")

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

    # Proof of GPU memory use
    mem_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
    print(f"\n[OK] GPU memory allocated: {mem_mb:.1f} MiB on {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    # Optional: make tokenizer quiet
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
