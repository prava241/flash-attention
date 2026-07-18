#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

#include "kernels.h"

using namespace std;

/*
 * Naive Kernels:
 * - Naive Matrix Multiplication (Regular + Transpose)
 * - Constant Division (for Normalization)
 * - Naive Softmax
 */

__global__ void matmulT_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    // Global row/column this thread is responsible for.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N)
        return;

    float sum = 0.0f;

    // Dot product of A's row and B's row.
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[col * K + k];
    }

    C[row * N + col] = sum;
}

__global__ void scale_kernel(
    float* C,
    float factor,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    C[i] *= factor;
}

__global__ void softmax_kernel(
    float* A,
    int N)
{
    __shared__ float reductions[256];
    float* row = A + blockIdx.x * N;

    int tid = threadIdx.x;
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = max(local_max, row[i]);
    }
    reductions[tid] = local_max;
    __syncthreads();

    // assume block size is a power of 2
    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if (tid < stride) {
            reductions[tid] = max(reductions[tid], reductions[tid + stride]);
        }
        __syncthreads();
    }

    float row_max = reductions[0];

    for (int i = tid; i < N; i += blockDim.x) {
        row[i] -= row_max;
        row[i] = expf(row[i]);
    }

    float local_sum = 0;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum = local_sum + row[i];
    }
    reductions[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if (tid < stride) {
            reductions[tid] = reductions[tid] + reductions[tid + stride];
        }
        __syncthreads();
    }

    float row_sum = reductions[0];

    for (int i = tid; i < N; i += blockDim.x) {
        row[i] /= row_sum;
    }
}

__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    // Global row/column this thread is responsible for.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N)
        return;

    float sum = 0.0f;

    // Dot product of A's row and B's column.
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

/* 
 * Tiled GEMM
 */

__global__ void tiled_mmT_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    int idxT = threadIdx.x * blockDim.x + threadIdx.y;

    __shared__ float A_shared[256];
    __shared__ float B_shared[256];

    float sum = 0.0f;

    for (int tile = 0; tile < (K + blockDim.x - 1) / blockDim.x; tile++) {

        int A_k = tile * blockDim.x + threadIdx.x;
        int B_k = tile * blockDim.x + threadIdx.y;

        if (row < M && A_k < K)
            A_shared[idx] = A[row*K + A_k];
        else
            A_shared[idx] = 0.0f;

        if (col < N && B_k < K)
            B_shared[idxT] = B[col*K + B_k];
        else
            B_shared[idxT] = 0.0f;

        __syncthreads();

        for (int k = 0; k < blockDim.x; k++) {
            sum +=
                A_shared[threadIdx.y * blockDim.x + k] *
                B_shared[threadIdx.x * blockDim.x + k];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row*N + col] = sum;
}

__global__ void tiled_mm_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ float A_shared[256];
    __shared__ float B_shared[256];

    float sum = 0.0f;

    for (int tile = 0; tile < (K + blockDim.x - 1) / blockDim.x; tile++) {

        int A_k = tile * blockDim.x + threadIdx.x;
        int B_k = tile * blockDim.x + threadIdx.y;

        if (row < M && A_k < K)
            A_shared[idx] = A[row*K + A_k];
        else
            A_shared[idx] = 0.0f;

        if (col < N && B_k < K)
            B_shared[idx] = B[B_k*N + col];
        else
            B_shared[idx] = 0.0f;

        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < blockDim.x; k++) {
            sum +=
                A_shared[threadIdx.y * blockDim.x + k] *
                B_shared[threadIdx.x + blockDim.x * k];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row*N + col] = sum;
}

__global__ void tiled_scaled_mmT_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float factor
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    int idxT = threadIdx.x * blockDim.x + threadIdx.y;

    __shared__ float A_shared[256];
    __shared__ float B_shared[256];

    float sum = 0.0f;

    for (int tile = 0; tile < (K + blockDim.x - 1) / blockDim.x; tile++) {

        int A_k = tile * blockDim.x + threadIdx.x;
        int B_k = tile * blockDim.x + threadIdx.y;

        if (row < M && A_k < K)
            A_shared[idx] = A[row*K + A_k];
        else
            A_shared[idx] = 0.0f;

        if (col < N && B_k < K)
            B_shared[idxT] = B[col*K + B_k];
        else
            B_shared[idxT] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < blockDim.x; k++) {
            sum +=
                A_shared[threadIdx.y * blockDim.x + k] *
                B_shared[threadIdx.x * blockDim.x + k];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row*N + col] = sum * factor;
}

__global__ void shfl_softmax_kernel(
    float* A,
    int N)
{
    __shared__ float reductions[8]; // TODO: dynamic sizing
    float* row = A + blockIdx.x * N;

    int tid = threadIdx.x;
    int warp = tid / warpSize;
    int lane = tid % warpSize;

    float local_max = -INFINITY;
    for (int j = 0, i = tid; i < N; ++j, i += blockDim.x) {
        local_max = max(local_max, row[i]);
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_max = max(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    if (lane == 0) reductions[warp] = local_max;
    __syncthreads();

    if (warp == 0) {
        float val = (lane < 8) ? reductions[lane] : -INFINITY;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val = max(val, __shfl_down_sync(0xffffffff, val, offset));
        }

        if (lane == 0)
            reductions[0] = val;
    }
    __syncthreads();

    float row_max = reductions[0];
    float local_sum = 0;

    for (int j = 0, i = tid; i < N; ++j, i += blockDim.x) {
        row[i] = expf(row[i] - row_max);
        local_sum += row[i];
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane == 0) reductions[warp] = local_sum;
    __syncthreads();

    if (warp == 0) {
        float val = (lane < 8) ? reductions[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (lane == 0)
            reductions[0] = val;
    }
    __syncthreads();

    float row_sum = reductions[0];
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < N; i += blockDim.x) {
        row[i] *= inv_sum;
    }
}