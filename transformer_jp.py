# gpu_check_japanese_hf.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


def assert_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is NOT available. Check nvidia-smi / driver / CUDA install.")
    print(f"[OK] CUDA available: {torch.version.cuda}, device count={torch.cuda.device_count()}")
    print(f"[OK] Using device 0: {torch.cuda.get_device_name(0)}")


def main():
    assert_cuda()
    device = torch.device("cuda:0")

    # Japanese BERT model
    model_id = "cl-tohoku/bert-base-japanese"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # FP16 to fit nicely on T4
        low_cpu_mem_usage=True,
    ).to(device).eval()

    # Double-check GPU usage
    p = next(model.parameters())
    assert p.is_cuda, "Model parameters are on CPU; expected CUDA."
    print(f"[OK] Model dtype={p.dtype}, device={p.device}")

    # Japanese input with [MASK] token for MLM
    prompt = "私は[MASK]が好きです。"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.topk(outputs.logits[0, inputs["input_ids"][0] == tokenizer.mask_token_id], k=5)

    tokens = [tokenizer.decode([idx]) for idx in predictions.indices[0]]
    print("\n--- Predictions for [MASK] ---")
    for tok, score in zip(tokens, predictions.values[0]):
        print(f"{tok} (score={score:.4f})")

    # Proof of GPU memory use
    mem_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
    print(f"\n[OK] GPU memory allocated: {mem_mb:.1f} MiB on {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
