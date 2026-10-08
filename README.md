CUDA LLM Optimization Lab

A hands-on CUDA performance engineering project focused on optimizing LLM inference.

The project starts with a reproducible PyTorch baseline using sshleifer/tiny-gpt2, profiles the real GPU workload, and then isolates expensive operations for custom CUDA optimization.

Project Goal

PyTorch LLM baseline
        ↓
GPU profiling
        ↓
Find bottlenecks
        ↓
Isolate expensive operation
        ↓
Custom CUDA kernel
        ↓
Optimize memory + execution
        ↓
Measure speedup

Later stages will extend this toward attention optimization, kernel fusion, and workload-specific autotuning/RL.

Project Layout

cuda-llm-optimization/
│
├── models/
│   └── tiny-gpt2/              # downloaded Hugging Face model files
│
├── phase1/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   └── benchmark.py
│
├── phase2/
│   ├── __init__.py
│   ├── profile.py
│   └── README.md
│
├── phase3/
│   ├── __init__.py
│   ├── inspect_mm_shapes.py
│   ├── benchmark_matmul.py
│   └── cuda_matmul.cu
│
├── scripts/
│   ├── download_model.py
│   └── smoke_test.py
│
├── .gitignore
├── requirements.txt
└── README.md

1. Create and Activate a Virtual Environment

Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

Linux/macOS:

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

2. Install PyTorch with CUDA

Install the CUDA-enabled PyTorch build appropriate for your GPU and driver from:

https://pytorch.org/get-started/locally/

Verify the installation:

python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('torch CUDA:', torch.version.cuda)"

The intended target is an NVIDIA GPU with CUDA support.

3. Install Project Dependencies

pip install -r requirements.txt

4. Download tiny-gpt2 Locally

python scripts/download_model.py

The model is stored under:

models/tiny-gpt2/

Phase 1 — PyTorch Baseline

Goal

Establish a reproducible inference baseline before writing custom CUDA kernels.

The model is loaded with Hugging Face Transformers and executed on the NVIDIA GPU.

Smoke Test

python scripts/smoke_test.py

This checks:

GPU availability

PyTorch CUDA availability

tiny-GPT2 loading

one real forward pass

input tensor shape

output logits shape

Baseline Benchmark

python -m phase1.benchmark --seq-len 128 --batch-size 1 --warmup 20 --iterations 100

Additional workloads:

python -m phase1.benchmark --seq-len 256 --batch-size 1 --warmup 20 --iterations 100
python -m phase1.benchmark --seq-len 128 --batch-size 4 --warmup 20 --iterations 100

Metrics

The benchmark records:

GPU name

model parameter count

input shape

GPU execution latency using CUDA events

end-to-end wall-clock latency

token-position throughput

peak GPU memory allocated

Phase 1 gives us the baseline to beat.

Phase 2 — GPU Profiling

Goal

Understand what actually happens inside:

model(input_ids)

Instead of treating the model as one black-box operation, PyTorch Profiler records the underlying operators and their GPU activity.

Run:

python -m phase2.profile --seq-len 128 --batch-size 1

The profiler records CPU and CUDA activity, tensor shapes, and memory information, then exports:

phase2/tiny_gpt2_trace.json

What We Found

For the current tiny-GPT2 workload on the RTX 3050, the profiler showed matrix multiplication as the largest individual GPU contribution, followed by LayerNorm and additional matrix-multiplication work.

Representative entries included:

aten::mm
    ↓
ampere_sgemm_128x64_tn

along with operations such as:

aten::addmm
aten::layer_norm
aten::bmm
aten::_softmax
aten::mul
aten::add

The important methodology is:

Do not guess the bottleneck.
Measure the workload first.

Phase 3 — Custom CUDA Matrix Multiplication

Goal

Isolate the profiled aten::mm operation and build a controlled custom CUDA GEMM benchmark.

We do not replace the GEMM inside the whole GPT-2 model yet.

The progression is:

Phase 2 profiler
      ↓
Find dominant aten::mm
      ↓
Extract exact matrix shapes
      ↓
Isolate GEMM
      ↓
Naive CUDA kernel
      ↓
Compare against torch.mm
      ↓
Optimize the CUDA kernel

Step 1 — Extract the Real GEMM Shape

Run:

python -m phase3.inspect_mm_shapes --seq-len 128 --batch-size 1

The current profiled workload produced:

A = [128 × 2]
B = [2 × 50257]
C = [128 × 50257]

Mathematically:

[128 × 2] × [2 × 50257]
          ↓
[128 × 50257]

The observed CUDA self time for this GEMM was approximately:

183.617 µs

This becomes our first controlled target.

Step 2 — Build and Benchmark the Naive CUDA Kernel

Example:

python -m phase3.benchmark_matmul --m 128 --k 2 --n 50257 --warmup 20 --iterations 100

The benchmark:

builds the CUDA extension from cuda_matmul.cu

allocates FP32 matrices on the GPU

computes a reference result using torch.mm

computes the result with our CUDA kernel

checks numerical correctness

measures both implementations with CUDA events

reports latency and approximate GFLOP/s

First Kernel Design

Each CUDA thread computes one output element:

C[row, col] = Σ A[row, k] × B[k, col]

The first implementation deliberately uses a simple 2D thread mapping and direct global-memory loads.

It is intentionally not optimized.

Optimization Roadmap

Naive CUDA GEMM
      ↓
Coalesced memory access
      ↓
Shared-memory tiling
      ↓
Register blocking
      ↓
Warp-level optimization
      ↓
Occupancy / launch tuning
      ↓
Compare with optimized PyTorch / cuBLAS

We expect the first custom kernel to be slower than PyTorch's highly optimized GEMM implementation. That is useful: it gives us a baseline for learning exactly which CUDA optimization techniques produce measurable improvements.

Current Hardware / Software Baseline

Current development environment:

GPU:          NVIDIA GeForce RTX 3050 6GB Laptop GPU
PyTorch:      2.11.0+cu128
PyTorch CUDA: 12.8
Model:        sshleifer/tiny-gpt2

Future Phases

Phase 4 → Optimize attention
Phase 5 → LayerNorm / elementwise optimization
Phase 6 → Kernel fusion + memory optimization
Phase 7 → End-to-end LLM inference optimization
Phase 8 → RL-based kernel autotuning

The final objective is to connect low-level CUDA optimization decisions to measurable LLM inference performance.