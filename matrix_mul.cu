#include <stdio.h>

#define N 2

__global__ void matrixMul(int *A, int *B, int *C) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    int sum = 0;
    for(int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

int main() {
    int A[N][N] = {{1, 2},
                   {3, 4}};

    int B[N][N] = {{5, 6},
                   {7, 8}};

    int C[N][N];

    int *dA, *dB, *dC;

    cudaMalloc(&dA, N*N*sizeof(int));
    cudaMalloc(&dB, N*N*sizeof(int));
    cudaMalloc(&dC, N*N*sizeof(int));

    cudaMemcpy(dA, A, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N*N*sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(N, N);
    dim3 blocks(1, 1);

    matrixMul<<<blocks, threads>>>(dA, dB, dC);

    cudaMemcpy(C, dC, N*N*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result matrix C:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
