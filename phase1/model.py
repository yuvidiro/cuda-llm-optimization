from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_dir: Path, device: torch.device):
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory does not exist: {model_dir}\n"
            "Run: python scripts/download_model.py"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float32,
    )

    model = model.to(device)
    model.eval()

    return tokenizer, model
