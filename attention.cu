#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

using namespace std;

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

__global__ void const_div(
    float* C,
    float factor,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    C[i] /= factor;
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