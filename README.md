A project I'm using to learn CUDA and rediscover my love of C/C++ performance engineering.

I am implementing the optimizations in FlashAttention from scratch. The end result is an optimized attention kernel with performance compared against a naive CUDA baseline. Correctness is verified against PyTorch.

## Optimizations implemented:

- [x] Naive CUDA attention implementation
  - Separate kernels for:
    - QKᵀ matrix multiplication
    - scaling
    - row-wise softmax
    - attention-value multiplication

- [x] Tiled GEMM
  - Use shared memory to improve matrix multiplication throughput
  - Reduce redundant global memory accesses

- [x] Optimized reductions
  - Warp-level reductions
  - Faster softmax max/sum computation

- [x] FlashAttention-style tiled attention
  - Implement streaming softmax
  - Blockwise attention using shared-memory tiles

- [ ] Optimized load/store patterns
  - Register tiling
  - Vectorized loads
  - Double buffering

- [ ] More efficient computation
  - Tensor cores for matrix multiplication

## Goals

- Understand GPU memory hierarchy and CUDA execution model
- Learn performance engineering techniques used in modern GPU kernels
- Recreate the core ideas behind FlashAttention from first principles

## Benchmarking

Performance is compared against:
- naive CUDA implementation
- PyTorch attention implementation

Correctness is checked using numerical comparisons against PyTorch outputs.

Current results:
- Naive PyTorch: ~2 ms
- Optimized PyTorch: ~1.6 ms
- Naive CUDA: ~25 ms
- Tiled CUDA: ~15 ms
- FlashAttention: ~10 ms (current best)