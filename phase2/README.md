# Phase 2 — GPU Profiling

Run this from the project root:

```powershell
python -m phase2.profile --seq-len 128 --batch-size 1
```

It will:

1. Load the same local tiny-gpt2 checkpoint used in Phase 1.
2. Run a short wait/warmup period.
3. Profile several inference iterations.
4. Print operators sorted by CUDA time.
5. Export `phase2/tiny_gpt2_trace.json`.

Phase 2 deliberately does not implement custom CUDA yet.

The workflow is:

```text
PyTorch baseline
      ↓
GPU profile
      ↓
find expensive operation
      ↓
inspect tensor shapes
      ↓
custom CUDA kernel
      ↓
benchmark speedup
```
