# hf_gpt_oss_20b_gpu_check.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "openai/gpt-oss-20b"


def assert_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is NOT available. Check nvidia-smi / driver / CUDA install.")
    print(f"[OK] CUDA available: {torch.version.cuda}, device count={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"    - device {i}: {torch.cuda.get_device_name(i)}")


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    assert_cuda()

    # T4 (16GB) works well; model weights include MXFP4 quant for MoE so it fits in ~16GB.
    # We'll let HF/Accelerate place things for us.
    device = torch.device("cuda:0")

    print(f"[INFO] Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    # Pad token alignment (safe for decoder-only LMs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading model: {MODEL_ID}")
    # dtype="auto" lets Transformers pick appropriate precision (bf16 weights + quantized experts).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # Double-check GPU usage
    p = next(model.parameters())
    assert p.is_cuda, "Model parameters are on CPU; expected CUDA."
    print(f"[OK] Model loaded on {p.device}, dtype={p.dtype}")

    # ---- Harmony chat format via chat template ----
    # The model expects Harmony-style messages; use the tokenizer's chat template.
    messages = [
        {"role": "user", "content": "以下の質問に日本語で簡潔に答えてください。日本食の特徴は何ですか？"}
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate on GPU
    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(
            prompt_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("\n--- Generation ---")
    print(text)

    # Proof of GPU memory use
    torch.cuda.synchronize()
    mem_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"\n[OK] GPU memory (current/peak): {mem_mb:.1f} / {peak_mb:.1f} MiB on {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
