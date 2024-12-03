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


void adi_column_update_host(int n, DATA_TYPE *X, DATA_TYPE *A, DATA_TYPE *B)
{
    for (int i1 = 0; i1 < n; i1++)
    {
        for (int i2 = 1; i2 < n; i2++)
        {
            
            X[i1 * n + i2] -= X[i1 * n + i2 - 1] * A[i1 * n + i2] / B[i1 * n + i2 - 1];
            B[i1 * n + i2] -= A[i1 * n + i2] * A[i1 * n + i2] / B[i1 * n + i2 - 1];
        }
    }

}

void adi_row_update_host(int n, DATA_TYPE *X, DATA_TYPE *A, DATA_TYPE *B)
{
    for (int i1 = 1; i1 < n; i1++)
        {
            for (int i2 = 0; i2 < n; i2++)
            {
                X[i1 * n + i2] -= X[(i1 - 1) * n + i2] * A[i1 * n + i2] / B[(i1 - 1) * n + i2];
                B[i1 * n + i2] -= A[i1 * n + i2] * A[i1 * n + i2] / B[(i1 - 1) * n + i2];
            }
        }
}

__global__ void adi_column_update(int n, DATA_TYPE *X, DATA_TYPE *A, DATA_TYPE *B)
{
    // Ogni thread gestisce una riga specifica
    int i1 = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i1 < n)
    {
        // Calcolo sequenziale lungo le colonne (i2)
        for (int i2 = 1; i2 < n; i2++)
        {
            X[i1 * n + i2] -= X[i1 * n + i2 - 1] * A[i1 * n + i2] / B[i1 * n + i2 - 1];
            B[i1 * n + i2] -= A[i1 * n + i2] * A[i1 * n + i2] / B[i1 * n + i2 - 1];
        }
    }
}


// Kernel per normalizzazione lungo l'ultima colonna
__global__ void adi_column_normalize(int n, DATA_TYPE *X, DATA_TYPE *B)
{
    int i1 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i1 < n)
    {
        X[i1 * n + (n - 1)] /= B[i1 * n + (n - 1)];
    }
}

// Kernel per back-substitution lungo le colonne
__global__ void adi_column_backsub(int n, DATA_TYPE *X, DATA_TYPE *A, DATA_TYPE *B)
{
    int i1 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i1 < n)
    {
        for (int i2 = 0; i2 < n - 2; i2++)
        {
            X[i1 * n + (n - i2 - 2)] = (X[i1 * n + (n - i2 - 2)] - 
                                        X[i1 * n + (n - i2 - 3)] * A[i1 * n + (n - i2 - 3)]) / 
                                        B[i1 * n + (n - i2 - 3)];
        }
    }
}


__global__ void adi_row_update(int n, DATA_TYPE *X, DATA_TYPE *A, DATA_TYPE *B)
{
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i2 < n)
    {
        for (int i1 = 1; i1 < n; i1++)
        {
            X[i1 * n + i2] -= X[(i1 - 1) * n + i2] * A[i1 * n + i2] / B[(i1 - 1) * n + i2];
            B[i1 * n + i2] -= A[i1 * n + i2] * A[i1 * n + i2] / B[(i1 - 1) * n + i2];
        }
    }
}

// Kernel per normalizzazione lungo l'ultima riga
__global__ void adi_row_normalize(int n, DATA_TYPE *X, DATA_TYPE *B)
{
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i2 < n)
    {
        X[(n - 1) * n + i2] /= B[(n - 1) * n + i2];
    }
}

// Kernel per back-substitution lungo le righe
__global__ void adi_row_backsub(int n, DATA_TYPE *X, DATA_TYPE *A, DATA_TYPE *B)
{
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i2 < n)
    {
        for (int i1 = 0; i1 < n - 2; i1++)
        {
            X[(n - 2 - i1) * n + i2] = (X[(n - 2 - i1) * n + i2] - 
                                        X[(n - i1 - 3) * n + i2] * A[(n - 3 - i1) * n + i2]) / 
                                        B[(n - 2 - i1) * n + i2];
        }
    }
}




void kernel_adi_device(int tsteps, int n, DATA_TYPE *d_X, DATA_TYPE *d_A, DATA_TYPE *d_B)
{
    // Configurazione per kernel con parallelismo lungo 1 dimensione
    dim3 dimBlock1D(BLOCK_SIZE);  // Numero di thread per blocco
    dim3 dimGrid1D((n + BLOCK_SIZE - 1) / BLOCK_SIZE);  // Numero di blocchi

    for (int t = 0; t < tsteps; t++)
    {
        // 1. Aggiornamento lungo le colonne
        adi_column_update<<<dimGrid1D, dimBlock1D>>>(n, d_X, d_A, d_B);
        cudaDeviceSynchronize();  // Sincronizza per completare l'aggiornamento

        // 2. Normalizzazione lungo l'ultima colonna
        adi_column_normalize<<<dimGrid1D, dimBlock1D>>>(n, d_X, d_B);
        cudaDeviceSynchronize();

        // 3. Back-substitution lungo le colonne
        adi_column_backsub<<<dimGrid1D, dimBlock1D>>>(n, d_X, d_A, d_B);
        cudaDeviceSynchronize();

        // 4. Aggiornamento lungo le righe
        adi_row_update<<<dimGrid1D, dimBlock1D>>>(n, d_X, d_A, d_B);
        cudaDeviceSynchronize();

        // 5. Normalizzazione lungo l'ultima riga
        adi_row_normalize<<<dimGrid1D, dimBlock1D>>>(n, d_X, d_B);
        cudaDeviceSynchronize();

        // 6. Back-substitution lungo le righe
        adi_row_backsub<<<dimGrid1D, dimBlock1D>>>(n, d_X, d_A, d_B);
        cudaDeviceSynchronize();
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
            if (fabs(X_host[i * n + j] - X_device[i * n + j]) > 1e-4)
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

    DATA_TYPE *X_host, *X_dev, *A, *B_host, *B_dev;

    // Allocazione memoria unificata
    cudaMallocManaged(&X_host, n * n * sizeof(DATA_TYPE));
    cudaMallocManaged(&X_dev, n * n * sizeof(DATA_TYPE));
    cudaMallocManaged(&A, n * n * sizeof(DATA_TYPE));
    cudaMallocManaged(&B_host, n * n * sizeof(DATA_TYPE));
    cudaMallocManaged(&B_dev, n * n * sizeof(DATA_TYPE));

    // Inizializzazione
    init_array(n, X_host, X_dev, A, B_host, B_dev);

    // Esecuzione su CPU
    clock_gettime(CLOCK_REALTIME, rt);
    kernel_adi_host(tsteps, n, X_host, A, B_host);
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("ADI (Host): %9.3f sec\n", wt);

    // Esecuzione su GPU
    clock_gettime(CLOCK_REALTIME, rt);
    kernel_adi_device(tsteps, n, X_dev, A, B_dev);
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("ADI (GPU): %9.3f sec\n", wt);

    // Confronto risultati
    if (compare_matrices(X_host, X_dev, n))
    {
        printf("Risultati Host e Device CORRETTI!\n");
    }
    else
    {
        printf("Risultati Host e Device NON corrispondono!\n");
    }

    // Liberazione memoria
    cudaFree(X_host);
    cudaFree(X_dev);
    cudaFree(A);
    cudaFree(B_host);
    cudaFree(B_dev);

    return 0;
}
