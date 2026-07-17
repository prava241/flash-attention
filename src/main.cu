#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "kernels.h"

int main()
{
    int M = 64;
    int K = 64;
    int N = 64;

    size_t bytesA = M*K*sizeof(float);
    size_t bytesB = K*N*sizeof(float);
    size_t bytesC = M*N*sizeof(float);

    std::vector<float> h_A(M*K, 1.0f);
    std::vector<float> h_B(K*N, 2.0f);
    std::vector<float> h_C(M*N);


    float *A,*B,*C;

    cudaMalloc(&A, bytesA);
    cudaMalloc(&B, bytesB);
    cudaMalloc(&C, bytesC);


    cudaMemcpy(A,h_A.data(),bytesA,cudaMemcpyHostToDevice);
    cudaMemcpy(B,h_B.data(),bytesB,cudaMemcpyHostToDevice);


    dim3 block(16,16);
    dim3 grid(
        (N+15)/16,
        (M+15)/16
    );


    matmul_kernel<<<grid,block>>>(
        A,B,C,M,N,K
    );


    cudaMemcpy(
        h_C.data(),
        C,
        bytesC,
        cudaMemcpyDeviceToHost
    );


    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            std::cout<<h_C[i*N+j]<<" ";
        }
        std::cout<<"\n";
    }


    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}