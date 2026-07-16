#include <attention.h>

int M = 512;
int K = 256;
int N = 1024;

float *d_A, *d_B, *d_C;

cudaMalloc(&d_A, M * K * sizeof(float));
cudaMalloc(&d_B, K * N * sizeof(float));
cudaMalloc(&d_C, M * N * sizeof(float));

// Copy host data to GPU...
cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

dim3 block(16, 16);
dim3 grid(
    (N + 15) / 16,
    (M + 15) / 16
);

matmul_kernel<<<grid, block>>>(
    d_A,
    d_B,
    d_C,
    M,
    N,
    K
);

cudaDeviceSynchronize();

cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);