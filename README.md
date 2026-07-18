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

- [ ] Kernel fusion
  - Fuse attention operations to reduce global memory traffic
  - Avoid unnecessary intermediate tensors

- [ ] Online softmax
  - Implement numerically stable streaming softmax
  - Maintain running statistics while processing attention tiles

- [ ] FlashAttention-style tiled attention
  - Avoid materializing the full attention matrix
  - Compute attention using shared-memory tiles

## Goals

- Understand GPU memory hierarchy and CUDA execution model
- Learn performance engineering techniques used in modern GPU kernels
- Recreate the core ideas behind FlashAttention from first principles

## Benchmarking

Performance is compared against:
- naive CUDA implementation
- PyTorch attention implementation

Correctness is checked using numerical comparisons against PyTorch outputs.