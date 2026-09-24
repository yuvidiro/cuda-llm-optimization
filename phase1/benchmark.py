from __future__ import annotations

import argparse
import time

import torch

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_ITERATIONS,
    DEFAULT_SEQ_LEN,
    DEFAULT_WARMUP,
    MODEL_DIR,
)
from .model import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 tiny-gpt2 CUDA baseline benchmark"
    )
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    return parser.parse_args()


def build_input(
    tokenizer,
    model,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    prompt = (
        "GPU performance engineering starts with measuring the baseline "
        "before changing the implementation."
    )

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = encoded["input_ids"]

    if input_ids.shape[1] > seq_len:
        input_ids = input_ids[:, :seq_len]
    elif input_ids.shape[1] < seq_len:
        pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = model.config.eos_token_id

        if pad_id is None:
            raise RuntimeError("Could not determine a padding token.")

        pad_count = seq_len - input_ids.shape[1]
        padding = torch.full(
            (1, pad_count),
            pad_id,
            dtype=input_ids.dtype,
        )
        input_ids = torch.cat([input_ids, padding], dim=1)

    input_ids = input_ids.repeat(batch_size, 1)
    return input_ids.to(device)


def cuda_event_benchmark(
    model,
    input_ids: torch.Tensor,
    warmup: int,
    iterations: int,
):
    for _ in range(warmup):
        with torch.inference_mode():
            _ = model(input_ids, use_cache=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    with torch.inference_mode():
        for _ in range(iterations):
            _ = model(input_ids, use_cache=True)

    end_event.record()
    torch.cuda.synchronize()

    total_gpu_ms = start_event.elapsed_time(end_event)
    avg_gpu_ms = total_gpu_ms / iterations

    return avg_gpu_ms


def wall_clock_benchmark(
    model,
    input_ids: torch.Tensor,
    iterations: int,
) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(iterations):
            _ = model(input_ids, use_cache=True)

    torch.cuda.synchronize()
    end = time.perf_counter()

    return ((end - start) * 1000.0) / iterations


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build "
            "and make sure an NVIDIA driver/GPU is available."
        )

    device = torch.device("cuda")

    print("=" * 72)
    print("Phase 1: tiny-gpt2 CUDA baseline")
    print("=" * 72)
    print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:          {torch.__version__}")
    print(f"PyTorch CUDA:     {torch.version.cuda}")
    print(f"Model directory:  {MODEL_DIR}")
    print()

    tokenizer, model = load_model(MODEL_DIR, device)
    input_ids = build_input(
        tokenizer,
        model,
        args.batch_size,
        args.seq_len,
        device,
    )

    parameter_count = sum(p.numel() for p in model.parameters())

    print(f"Parameters:       {parameter_count:,}")
    print(f"Input shape:      {tuple(input_ids.shape)}")
    print(f"Dtype:            {next(model.parameters()).dtype}")
    print()

    # Smoke-test one forward pass before timing.
    with torch.inference_mode():
        outputs = model(input_ids, use_cache=True)

    print(f"Logits shape:     {tuple(outputs.logits.shape)}")
    print()

    gpu_latency_ms = cuda_event_benchmark(
        model,
        input_ids,
        args.warmup,
        args.iterations,
    )

    wall_latency_ms = wall_clock_benchmark(
        model,
        input_ids,
        args.iterations,
    )

    peak_allocated_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    )

    tokens_per_iteration = args.batch_size * args.seq_len
    gpu_tokens_per_sec = tokens_per_iteration / (gpu_latency_ms / 1000.0)
    wall_tokens_per_sec = tokens_per_iteration / (wall_latency_ms / 1000.0)

    print("Results")
    print("-" * 72)
    print(f"GPU execution latency: {gpu_latency_ms:.4f} ms")
    print(f"Wall-clock latency:    {wall_latency_ms:.4f} ms")
    print(f"GPU token throughput:  {gpu_tokens_per_sec:,.2f} tokens/s")
    print(f"Wall token throughput: {wall_tokens_per_sec:,.2f} tokens/s")
    print(f"Peak allocated memory: {peak_allocated_mb:.2f} MiB")
    print("-" * 72)

    print()
    print("Save these numbers. They are our Phase 1 baseline.")


if __name__ == "__main__":
    main()
