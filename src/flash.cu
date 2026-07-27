#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

#include "kernels.h"
#include "utils.h"

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <N> <D>\n";
        return 1;
    }

    int N = std::atoi(argv[1]);
    int D = std::atoi(argv[2]);

    if (N <= 0 || D <= 0) {
        std::cerr << "Error: N and D must be positive integers.\n";
        return 1;
    }

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

    // one warp (blockDim.x == 32) per query row, blockDim.y rows per block
    dim3 flash_block(32, 8);
    dim3 flash_grid(
        1,
        (N + flash_block.y - 1) / flash_block.y
    );

    flash_attention<<<flash_grid,flash_block>>>(
        Q, K, V, N, D, O
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "flash_attention runtime: "
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
