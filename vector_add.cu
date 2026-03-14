#include <stdio.h>

#define N 4   // Size of vector

__global__ void vectorAdd(int *A, int *B, int *C) {
    int i = threadIdx.x;   // Each thread handles one element

    C[i] = A[i] + B[i];
}

int main() {

    int A[N] = {1, 2, 3, 4};
    int B[N] = {5, 6, 7, 8};
    int C[N];

    int *dA, *dB, *dC;

    // Allocate GPU memory
    cudaMalloc(&dA, N * sizeof(int));
    cudaMalloc(&dB, N * sizeof(int));
    cudaMalloc(&dC, N * sizeof(int));

    // Copy data from CPU to GPU
    cudaMemcpy(dA, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(N);   // N threads in x-direction
    dim3 blocks(1);    // 1 block

    // Launch kernel
    vectorAdd<<<blocks, threads>>>(dA, dB, dC);

    // Copy result back to CPU
    cudaMemcpy(C, dC, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result vector C:\n");
    for(int i = 0; i < N; i++) {
        printf("%d ", C[i]);
    }

    // Free GPU memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}