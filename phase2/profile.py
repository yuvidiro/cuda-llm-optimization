from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from phase1.config import MODEL_DIR
from phase1.model import load_model
from phase1.benchmark import build_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: profile tiny-gpt2 GPU execution"
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--wait", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--active", type=int, default=5)
    parser.add_argument("--output", type=str, default="phase2/tiny_gpt2_trace.json")
    parser.add_argument("--row-limit", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 2: tiny-gpt2 GPU profiling")
    print("=" * 80)
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print()

    tokenizer, model = load_model(MODEL_DIR, device)

    input_ids = build_input(
        tokenizer,
        model,
        args.batch_size,
        args.seq_len,
        device,
    )

    prof_schedule = schedule(
        wait=args.wait,
        warmup=args.warmup,
        active=args.active,
        repeat=1,
    )

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(args.wait + args.warmup + args.active):
            with record_function("tiny_gpt2_forward"):
                with torch.inference_mode():
                    _ = model(input_ids, use_cache=True)
            prof.step()

    print("Top operators by total CUDA time:")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=args.row_limit,
        )
    )

    print("\nTop operators by self CUDA time:")
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=args.row_limit,
        )
    )

    prof.export_chrome_trace(str(output_path))

    print(f"\nChrome trace written to: {output_path}")
    print("Open the JSON trace in a compatible Chrome/Perfetto trace viewer.")


if __name__ == "__main__":
    main()
