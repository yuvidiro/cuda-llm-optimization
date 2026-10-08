from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from phase1.benchmark import build_input
from phase1.config import MODEL_DIR
from phase1.model import load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find aten::mm input shapes in tiny-GPT2")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")
    tokenizer, model = load_model(MODEL_DIR, device)
    input_ids = build_input(tokenizer, model, args.batch_size, args.seq_len, device)

    # One profiled forward is enough to discover the shapes. This is not a
    # performance benchmark; profiling overhead is intentional here.
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
    ) as prof:
        with record_function("tiny_gpt2_shape_probe"):
            with torch.inference_mode():
                _ = model(input_ids, use_cache=True)

    events = [e for e in prof.events() if e.name == "aten::mm"]
    events.sort(key=lambda e: e.self_device_time_total, reverse=True)

    print("=" * 80)
    print("Phase 3: target GEMM shape probe")
    print("=" * 80)
    print(f"GPU:       {torch.cuda.get_device_name(0)}")
    print(f"Input:     {tuple(input_ids.shape)}")
    print(f"aten::mm calls observed: {len(events)}")
    print()

    for i, event in enumerate(events, start=1):
        print(f"MM #{i}")
        print(f"  CUDA self time: {event.self_device_time_total:.3f} us")
        print(f"  Input shapes:   {event.input_shapes}")
        print()

    if not events:
        print("No aten::mm event was found in this profiling pass.")
        print("Inspect aten::addmm as the next candidate.")
        raise SystemExit(1)

    print("Use the largest-CUDA-time MM shape as our first controlled target.")


if __name__ == "__main__":
    main()
