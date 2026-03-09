#include <stdio.h>
#include <stdlib.h>

#define ROWS 4
#define COLS 4

__global__ void matSub2D(float *A, float *B, float *C, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] - B[idx];   
    }
}

void printMatrix(const char *name, float *M, int rows, int cols)
{
    printf("%s:\n", name);
    for (int r = 0; r < rows; r++) {
        printf("  [ ");
        for (int c = 0; c < cols; c++)
            printf("%5.1f%s", M[r*cols+c], c<cols-1?" ":"");
        printf(" ]\n");
    }
    printf("\n");
}

int main(void)
{
    printf("\n===== 2D MATRIX SUBTRACTION  C = A - B =====\n\n");

    int    N    = ROWS * COLS;
    size_t size = N * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);


    for (int i = 0; i < N; i++) { h_A[i] = (float)((i+1)*10); h_B[i] = (float)(i+1); }

    printMatrix("A", h_A, ROWS, COLS);
    printMatrix("B", h_B, ROWS, COLS);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (COLS + blockSize.x - 1) / blockSize.x,
        (ROWS + blockSize.y - 1) / blockSize.y
    );

    matSub2D<<<gridSize, blockSize>>>(d_A, d_B, d_C, ROWS, COLS);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA Error: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printMatrix("C = A - B", h_C, ROWS, COLS);

    int ok = 1;
    for (int i = 0; i < N; i++)
        if (h_C[i] != h_A[i] - h_B[i]) { ok = 0; break; }
    printf("RESULT: %s\n", ok ? "ALL CORRECT!" : "ERRORS FOUND!");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
