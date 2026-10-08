from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT / "phase3" / "cuda_matmul.cu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare torch.mm against naive CUDA matmul")
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    return p.parse_args()


def benchmark(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for _ in range(iterations):
        _ = fn()

    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("=" * 80)
    print("Phase 3: controlled CUDA GEMM experiment")
    print("=" * 80)
    print(f"GPU:        {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:    {torch.__version__}")
    print(f"Torch CUDA: {torch.version.cuda}")
    print(f"Shape:      A[{args.m} x {args.k}] @ B[{args.k} x {args.n}]")
    print()

    print("Building CUDA extension (first run can take a little while)...")
    ext = load(
        name="phase3_naive_matmul",
        sources=[str(CUDA_SOURCE)],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )

    torch.manual_seed(1234)
    A = torch.randn(args.m, args.k, device="cuda", dtype=torch.float32)
    B = torch.randn(args.k, args.n, device="cuda", dtype=torch.float32)

    reference = torch.mm(A, B)
    custom = ext.matmul_cuda(A, B)
    torch.cuda.synchronize()

    max_error = (reference - custom).abs().max().item()
    mean_error = (reference - custom).abs().mean().item()

    print()
    print("Correctness")
    print("-" * 80)
    print(f"Max absolute error:  {max_error:.6e}")
    print(f"Mean absolute error: {mean_error:.6e}")
    print(f"torch.allclose:      {torch.allclose(reference, custom, rtol=1e-3, atol=1e-4)}")

    torch_ms = benchmark(
        lambda: torch.mm(A, B), args.warmup, args.iterations
    )
    custom_ms = benchmark(
        lambda: ext.matmul_cuda(A, B), args.warmup, args.iterations
    )

    flops = 2.0 * args.m * args.n * args.k
    torch_gflops = flops / (torch_ms * 1e6)
    custom_gflops = flops / (custom_ms * 1e6)

    print()
    print("Performance")
    print("-" * 80)
    print(f"PyTorch torch.mm:      {torch_ms:.4f} ms  ({torch_gflops:.2f} GFLOP/s)")
    print(f"Naive custom CUDA:     {custom_ms:.4f} ms  ({custom_gflops:.2f} GFLOP/s)")
    print(f"Custom / PyTorch:      {custom_ms / torch_ms:.2f}x latency")
    print()
    print("This is intentionally a naive kernel. It is our Phase 3 starting point, not the optimized result.")


if __name__ == "__main__":
    main()
