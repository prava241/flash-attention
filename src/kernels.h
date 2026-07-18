#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

__global__ void matmulT_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);

__global__ void scale_kernel(
    float* C,
    float factor,
    int n
);

__global__ void softmax_kernel(
    float* A,
    int N
);

__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);

__global__ void tiled_mmT_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);

__global__ void tiled_mm_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);

#endif // KERNELS_H