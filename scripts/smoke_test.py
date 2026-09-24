from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "tiny-gpt2"


def main() -> None:
    print("=" * 72)
    print("tiny-gpt2 smoke test")
    print("=" * 72)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check your PyTorch CUDA installation."
        )

    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"PyTorch CUDA: {torch.version.cuda}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.float32,
    ).cuda()
    model.eval()

    prompt = "CUDA optimization begins with"
    batch = tokenizer(prompt, return_tensors="pt")
    input_ids = batch["input_ids"].cuda()

    with torch.inference_mode():
        output = model(input_ids)

    print(f"Input shape:  {tuple(input_ids.shape)}")
    print(f"Logits shape: {tuple(output.logits.shape)}")
    print("Smoke test:   PASS")


if __name__ == "__main__":
    main()
