#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024
#define TILE_SIZE 16
#define BLOCK_ROWS 8

void matrixTransposeCPU(int* a, int* b)
{
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col)
        {
            b[row * N + col] = a[col * N + row];
        }
}

__global__ void matrixTransposeGPU(int* a, int* b)
{
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col)
        {
            b[row * N + col] = a[col * N + row];
        }
}

__global__ void matrixTransposeGPUOptimized(int* a, int* b)
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int width = gridDim.x * TILE_SIZE;

    for (int j = 0; j < TILE_SIZE; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = a[(y + j) * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_SIZE + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    for (int j = 0; j < TILE_SIZE; j += BLOCK_ROWS)
        b[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

void printMatrix(int* a)
{
    for (int row = 0; row < N; ++row)
    {
        for (int col = 0; col < N; ++col)
            printf("%d ", a[row * N + col]);
        printf("\n");
    }
}

int main()
{
    // Timing variables
    clock_t start, end;
    double cpu_time_used;

    // Start timer
    start = clock();

    int* a, * b;

    int size = N * N * sizeof(int); // Number of bytes of an N x N matrix

    // Allocate memory using standard malloc
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);

    // Initialize memory
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col)
        {
            a[row * N + col] = row;
            b[row * N + col] = 0; 
        }

    //printMatrix(a);

    dim3 dimGrid(N / TILE_SIZE, N / TILE_SIZE, 1);
    dim3 dimBlock(TILE_SIZE, BLOCK_ROWS, 1);

    int deviceId;
    cudaGetDevice(&deviceId);
    cudaMemPrefetchAsync(a, size, deviceId);
    cudaMemPrefetchAsync(b, size, deviceId);

    //matrixTransposeGPU<<<dimGrid, dimBlock>>>(a, b);
    matrixTransposeGPUOptimized<<<dimGrid, dimBlock>>>(a, b);

    cudaDeviceSynchronize();

    //matrixTransposeCPU(a, b);

    printf("Calculation completed!\n");

    //printMatrix(b);

    cudaFree(b);

    // End timer
    end = clock();

    // Calculate time taken in seconds
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    int* c;

    cudaMallocManaged(&c, size);

    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col)
            c[row * N + col] = 0;

    matrixTransposeCPU(a, c);

    //printMatrix(c);

    bool error = false;
    for (int row = 0; row < N && !error; ++row)
        for (int col = 0; col < N && !error; ++col)
            if (b[row * N + col] != c[row * N + col])
            {
                printf("FOUND ERROR at c[%d][%d]\n", row, col);
                error = true;
                break;
            }
    if (!error)
    {
        printf("Success!\n");
    }

    cudaFree(a);
    cudaFree(c);

    // Print timing information
    printf("Time taken: %f seconds\n", cpu_time_used);
    printf("Time taken: %f milliseconds\n", cpu_time_used * 1000);

    return 0;
}