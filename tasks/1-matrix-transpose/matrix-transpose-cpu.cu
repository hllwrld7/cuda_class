#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024

void matrixTransposeCPU(int *a, int *b)
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

void printMatrix(int *a)
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
    a = (int*)malloc(size);
    b = (int*)malloc(size);

    // Initialize memory
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col)
        {
            a[row * N + col] = row;
            b[row * N + col] = 0;
        }  

    printf("\n");

    //printMatrix(a);

    matrixTransposeCPU(a, b);

    // Verify the results (optional)
    printf("Calculation completed!\n");

    //printMatrix(b);

    // Free allocated memory
    free(a);
    free(b);

    // End timer
    end = clock();

    // Calculate time taken in seconds
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Print timing information
    printf("Time taken: %f seconds\n", cpu_time_used);
    printf("Time taken: %f milliseconds\n", cpu_time_used * 1000);

    return 0;
}