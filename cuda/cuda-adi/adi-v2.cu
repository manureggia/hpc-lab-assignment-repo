#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "adi.h"


#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}


void init_array(int n, DATA_TYPE *X, DATA_TYPE *X_DEV, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *B_DEV)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            X[i * n + j] = ((DATA_TYPE)i * (j + 1) + 1) / n;
            X_DEV[i * n + j] = ((DATA_TYPE)i * (j + 1) + 1) / n;
            A[i * n + j] = ((DATA_TYPE)i * (j + 2) + 2) / n;
            B[i * n + j] = ((DATA_TYPE)i * (j + 3) + 3) / n;
            B_DEV[i * n + j] = ((DATA_TYPE)i * (j + 3) + 3) / n;
        }
    }
}

void print_array(int n, DATA_TYPE *X)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%0.2f", X[i * n + j]);
            if ((i * n + j) % n == 0)
                printf("\n");
        }
    }
    printf("\n");
}

// Kernel host
void kernel_adi_host(int tsteps, int n, DATA_TYPE *X, DATA_TYPE *A, DATA_TYPE *B)
{
    for (int t = 0; t < tsteps; t++)
    {
        // Aggiornamento lungo le colonne
        for (int i1 = 0; i1 < n; i1++)
        {
            for (int i2 = 1; i2 < n; i2++)
            {
                X[i1 * n + i2] -= X[i1 * n + i2 - 1] * A[i1 * n + i2] / B[i1 * n + i2 - 1];
                B[i1 * n + i2] -= A[i1 * n + i2] * A[i1 * n + i2] / B[i1 * n + i2 - 1];
            }
        }

        // Normalizzazione
        for (int i1 = 0; i1 < n; i1++)
            X[i1 * n + (n - 1)] /= B[i1 * n + (n - 1)];

        // Back-substitution
        for (int i1 = 0; i1 < n; i1++)
        {
            for (int i2 = 0; i2 < n - 2; i2++)
                X[i1 * n + (n - i2 - 2)] = (X[i1 * n + (n - i2 - 2)] - X[i1 * n + (n - i2 - 3)] * A[i1 * n + (n - i2 - 3)]) / B[i1 * n + (n - i2 - 3)];
        }

        // Aggiornamento lungo le righe
        for (int i1 = 1; i1 < n; i1++)
        {
            for (int i2 = 0; i2 < n; i2++)
            {
                X[i1 * n + i2] -= X[(i1 - 1) * n + i2] * A[i1 * n + i2] / B[(i1 - 1) * n + i2];
                B[i1 * n + i2] -= A[i1 * n + i2] * A[i1 * n + i2] / B[(i1 - 1) * n + i2];
            }
        }

        // Normalizzazione
        for (int i2 = 0; i2 < n; i2++)
            X[(n - 1) * n + i2] /= B[(n - 1) * n + i2];

        // Back-substitution
        for (int i1 = 0; i1 < n - 2; i1++)
        {
            for (int i2 = 0; i2 < n; i2++)
                X[(n - 2 - i1) * n + i2] = (X[(n - 2 - i1) * n + i2] - X[(n - i1 - 3) * n + i2] * A[(n - 3 - i1) * n + i2]) / B[(n - 2 - i1) * n + i2];
        }
    }
}

//Kernel Cuda
__global__ void kernel_adi_device(int tsteps, int n, DATA_TYPE *X, DATA_TYPE *A, DATA_TYPE *B)
{
    int i1 = blockIdx.x * blockDim.x + threadIdx.x;
    int i2 = blockIdx.y * blockDim.y + threadIdx.y;

    for (int t = 0; t < tsteps; t++)
    {
        // Aggiornamento lungo le colonne
        if (i1 < n && i2 > 0 && i2 < n)
        {
            X[i1 * n + i2] -= X[i1 * n + i2 - 1] * A[i1 * n + i2] / B[i1 * n + i2 - 1];
            B[i1 * n + i2] -= A[i1 * n + i2] * A[i1 * n + i2] / B[i1 * n + i2 - 1];
        }
        __syncthreads();

        // Normalizzazione lungo l'ultima colonna
        if (i1 < n && i2 == n - 1)
        {
            X[i1 * n + i2] /= B[i1 * n + i2];
        }
        __syncthreads();

        // Back-substitution lungo le colonne
        if (i1 < n && i2 < n - 2)
        {
            X[i1 * n + (n - i2 - 2)] = (X[i1 * n + (n - i2 - 2)] - X[i1 * n + (n - i2 - 3)] * A[i1 * n + (n - i2 - 3)]) / B[i1 * n + (n - i2 - 3)];
                                    
        }
        __syncthreads();

        // Aggiornamento lungo le righe
        if (i1 > 0 && i1 < n && i2 < n)
        {
            X[i1 * n + i2] -= X[(i1 - 1) * n + i2] * A[i1 * n + i2] / B[(i1 - 1) * n + i2];
            B[i1 * n + i2] -= A[i1 * n + i2] * A[i1 * n + i2] / B[(i1 - 1) * n + i2];
        }
        __syncthreads();

        // Normalizzazione lungo l'ultima riga
        if (i1 == n - 1 && i2 < n)
        {
            X[i1 * n + i2] /= B[i1 * n + i2];
        }
        __syncthreads();

        // Back-substitution lungo le righe
        if (i1 < n - 2 && i2 < n)
        {
            X[(n - 2 - i1) * n + i2] = (X[(n - 2 - i1) * n + i2] - 
                                        X[(n - i1 - 3) * n + i2] * A[(n - 3 - i1) * n + i2]) / 
                                        B[(n - 2 - i1) * n + i2];
        }
        __syncthreads();
    }
}

// Confronta due matrici per verificare la correttezza
int compare_matrices(DATA_TYPE *X_host, DATA_TYPE *X_device, int n)
{
    int return_value = 1;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (fabs(X_host[i * n + j] - X_device[i * n + j]) > 1e-6)
            {
                printf("Mismatch at (%d, %d): Host = %f, Device = %f\n", i, j, X_host[i * n + j], X_device[i * n + j]);
                return_value = 0;
            }
        }
    }
    return return_value;
}

int main()
{
    int n = N;
    int tsteps = TSTEPS;
    double wt;
    struct timespec rt[2];

    DATA_TYPE *X, *X_dev, *A, *B, *B_dev;
    DATA_TYPE *d_X, *d_A, *d_B;

    X = (DATA_TYPE *)malloc(n * n * sizeof(DATA_TYPE));
    X_dev = (DATA_TYPE *)malloc(n * n * sizeof(DATA_TYPE));
    A = (DATA_TYPE *)malloc(n * n * sizeof(DATA_TYPE));
    B = (DATA_TYPE *)malloc(n * n * sizeof(DATA_TYPE));
    B_dev = (DATA_TYPE *)malloc(n * n * sizeof(DATA_TYPE));

    init_array(n, X, X_dev, A, B, B_dev);
    // Kernel host
    clock_gettime(CLOCK_REALTIME, rt);
    kernel_adi_host(tsteps, n, X, A, B);
    clock_gettime(CLOCK_REALTIME, rt + 1);

    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("ADI (Host): %9.3f sec\n", wt);

    // Allocazione memoria GPU
    cudaMalloc((void **)&d_X, sizeof(DATA_TYPE) * n * n);
    cudaMalloc((void **)&d_A, sizeof(DATA_TYPE) * n * n);
    cudaMalloc((void **)&d_B,sizeof(DATA_TYPE) * n * n);
    // Copia dei dati
    cudaMemcpy(d_X, X_dev, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_dev, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice);

    //Dimensionamento Griglia e Blocco (16 - con 32 non parte)
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n + (BLOCK_SIZE)-1) / (BLOCK_SIZE), (n + (BLOCK_SIZE)-1) / (BLOCK_SIZE));

    clock_gettime(CLOCK_REALTIME, rt);
    kernel_adi_device<<<dimGrid, dimBlock>>>(tsteps, n, d_X, d_A, d_B);
    gpuErrchk(cudaPeekAtLastError());  
    gpuErrchk(cudaDeviceSynchronize());
    clock_gettime(CLOCK_REALTIME, rt + 1);

    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("ADI (GPU): %9.3f sec\n", wt);
    cudaMemcpy(X_dev, d_X, n * n * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
/*
    if (compare_matrices(X, X_dev, n))
    {
        printf("Risultati Host e Device CORRETTI!\n");
    }
    else
    {
        printf("Risultati Host e Device NON corrispondono!\n");
    }
*/
    // Liberazione memoria
    free(X);
    free(X_dev);
    free(A);
    free(B_dev);
    free(B);
    cudaFree(d_X);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
