# CUDA LLM Optimization Lab

Phase 1 establishes a reproducible PyTorch baseline for `sshleifer/tiny-gpt2`.
The goal is to measure model inference before writing any custom CUDA kernels.

## Project layout

```text
cuda-llm-optimization/
├── models/
│   └── tiny-gpt2/          # downloaded Hugging Face model files
├── phase1/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   └── benchmark.py
├── scripts/
│   ├── download_model.py
│   └── smoke_test.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2. Install PyTorch with CUDA

Install the CUDA-enabled PyTorch build appropriate for your GPU/driver from:

https://pytorch.org/get-started/locally/

Then verify:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('torch CUDA:', torch.version.cuda)"
```

The project requires an NVIDIA GPU for the intended CUDA benchmark.

## 3. Install project dependencies

```bash
pip install -r requirements.txt
```

## 4. Download tiny-gpt2 locally

```bash
python scripts/download_model.py
```

The model is stored under:

```text
models/tiny-gpt2/
```

## 5. Run the smoke test

```bash
python scripts/smoke_test.py
```

This checks that the tokenizer, model, GPU, and one forward pass all work.

## 6. Run the Phase 1 baseline benchmark

```bash
python -m phase1.benchmark --seq-len 128 --batch-size 1 --warmup 20 --iterations 100
```

You can also try:

```bash
python -m phase1.benchmark --seq-len 256 --batch-size 1 --warmup 20 --iterations 100
python -m phase1.benchmark --seq-len 128 --batch-size 4 --warmup 20 --iterations 100
```

## What the benchmark reports

- GPU name
- model parameter count
- input shape
- GPU execution latency using CUDA events
- end-to-end wall-clock latency
- approximate token throughput
- peak GPU memory allocated

## Why Phase 1 is intentionally simple

Do not optimize anything yet.

We want a baseline that we can profile and compare against later:

```text
PyTorch baseline
       ↓
Nsight profiling
       ↓
Identify bottleneck
       ↓
Custom CUDA kernel
       ↓
Measure speedup
```

Later phases will introduce custom CUDA kernels, memory coalescing, shared-memory tiling, kernel fusion, and eventually autotuning/RL.
