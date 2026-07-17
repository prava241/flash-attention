#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstddef>
#include <cmath>
#include <chrono>

#include "kernels.h"

float* read_binary(
    const char* filename,
    size_t size
)
{
    float* data = new float[size];

    std::ifstream file(filename, std::ios::binary);

    file.read(
        reinterpret_cast<char*>(data),
        size * sizeof(float)
    );

    file.close();

    return data;
}

void write_binary(
    const char* filename,
    float* data,
    size_t size
)
{
    std::ofstream file(
        filename,
        std::ios::binary
    );

    file.write(
        reinterpret_cast<char*>(data),
        size*sizeof(float)
    );

    file.close();
}

void baseline_attention(
    float* Q,
    float* K,
    float* V,
    int N,
    int D,
    float* O
) 
{
    float* S;
    int S_elements = N*N;
    cudaMalloc(&S, S_elements*sizeof(float));

    dim3 mmt_block(16,16);
    dim3 mmt_grid(
        (N+15)/16,
        (N+15)/16
    );

    matmulT_kernel<<<mmt_grid,mmt_block>>>(
        Q, K, S, N, N, D
    );

    dim3 d_block(256);
    dim3 d_grid(
        (S_elements+255)/256
    );

    const_div<<<d_grid,d_block>>>(
        S, sqrtf((float)D), S_elements
    );

    dim3 s_block(256);
    dim3 s_grid(N);

    softmax_kernel<<<s_grid,s_block>>>(
        S, N
    );

    dim3 mm_block(16,16);
    dim3 mm_grid(
        (D+15)/16,
        (N+15)/16
    );

    matmul_kernel<<<mm_grid,mm_block>>>(
        S, V, O, N, D, N
    );

    cudaFree(S);
}

int main()
{
    int N = 512;
    int D = 64;

    size_t elements = N*D;
    size_t bytes = elements*sizeof(float);

    /*** Allocating and Loading Data ***/
    float *Q,*K,*V,*O;

    float* h_Q = read_binary(
        "data/q.bin",
        elements
    );
    cudaMalloc(&Q, bytes);
    cudaMemcpy(Q,h_Q,bytes,cudaMemcpyHostToDevice);

    float* h_K = read_binary(
        "data/k.bin",
        elements
    );
    cudaMalloc(&K, bytes);
    cudaMemcpy(K,h_K,bytes,cudaMemcpyHostToDevice);

    float* h_V = read_binary(
        "data/v.bin",
        elements
    );
    cudaMalloc(&V, bytes);
    cudaMemcpy(V,h_V,bytes,cudaMemcpyHostToDevice);

    float* h_O = new float[elements];
    cudaMalloc(&O, bytes);

    /*** Running and Timing GPU Computation ***/
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    baseline_attention(Q, K, V, N, D, O);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "baseline_attention runtime: "
            << milliseconds
            << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_O,O,bytes,cudaMemcpyDeviceToHost);

    /*** Writing Data, Freeing Memory ***/
    write_binary(
        "data/output.bin",
        h_O,
        elements
    );

    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O;

    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(O);
}