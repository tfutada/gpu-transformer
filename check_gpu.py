import os
import torch


def assert_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is NOT available. Check nvidia-smi / driver / CUDA install.")
    print(f"[OK] CUDA available: {torch.version.cuda}, device count={torch.cuda.device_count()}")
    print(f"[OK] Using device 0: {torch.cuda.get_device_name(0)}")


def main():
    assert_cuda()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
