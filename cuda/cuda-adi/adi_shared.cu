#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "adi.h"

#define gpuErrchk(ans)                  \
{                                       \
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

void print_array(int n, DATA_TYPE *X)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%0.2f ", X[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int compare_matrices(DATA_TYPE *X, DATA_TYPE *X_copy, int n)
{
    int return_value = 1;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (fabs(X[i * n + j] - X_copy[i * n + j]) > 1e-6)
            {
                printf("Mismatch at (%d, %d): Host = %f, Device = %f\n", i, j, X[i * n + j], X_copy[i * n + j]);
                return 0;
            }
        }
    }
    return 1;
}

// Host-side ADI
void host_adi(int tsteps, int n, DATA_TYPE *X, const DATA_TYPE *A, DATA_TYPE *B)
{
    for (int t = 0; t < tsteps; t++)
    {
        for (int row = 0; row < n; row++)
        {
            for (int col = 1; col < n; col++)
            {
                int idx = row * n + col;
                int prev_idx = row * n + (col - 1);
                if (B[prev_idx] != 0.f)
                {
                    X[idx] -= X[prev_idx] * A[idx] / B[prev_idx];
                    B[idx] -= A[idx] * A[idx] / B[prev_idx];
                }
            }
            int last_col = row * n + (n - 1);
            if (B[last_col] != 0.f)
                X[last_col] /= B[last_col];
            for (int col = n - 2; col >= 0; col--)
            {
                int idx = row * n + col;
                int next_idx = row * n + (col + 1);
                if (B[idx] != 0.f)
                    X[idx] = (X[idx] - X[next_idx] * A[next_idx]) / B[idx];
            }
        }

        for (int col = 0; col < n; col++)
        {
            for (int row = 1; row < n; row++)
            {
                int idx = row * n + col;
                int prev_idx = (row - 1) * n + col;
                if (B[prev_idx] != 0.f)
                {
                    X[idx] -= X[prev_idx] * A[idx] / B[prev_idx];
                    B[idx] -= A[idx] * A[idx] / B[prev_idx];
                }
            }
            int last_row = (n - 1) * n + col;
            if (B[last_row] != 0.f)
                X[last_row] /= B[last_row];
            for (int row = n - 2; row >= 0; row--)
            {
                int idx = row * n + col;
                int next_idx = (row + 1) * n + col;
                if (B[idx] != 0.f)
                    X[idx] = (X[idx] - X[next_idx] * A[next_idx]) / B[idx];
            }
        }
    }
}

// Shared memory kernels
__global__ void kernel_column_forward_elimination_shared(int n, DATA_TYPE *X, const DATA_TYPE *A, DATA_TYPE *B)
{
    __shared__ DATA_TYPE X_s[BLOCK_SIZE];
    __shared__ DATA_TYPE B_s[BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = threadIdx.x;

    if (row < n)
    {
        if (col < n)
        {
            X_s[col] = X[row * n + col];
            B_s[col] = B[row * n + col];
        }
        __syncthreads();

        for (int k = 1; k < n; k++)
        {
            if (col == k && B_s[k - 1] != 0.f)
            {
                X_s[k] -= X_s[k - 1] * A[row * n + k] / B_s[k - 1];
                B_s[k] -= A[row * n + k] * A[row * n + k] / B_s[k - 1];
            }
            __syncthreads();
        }

        if (col < n)
        {
            X[row * n + col] = X_s[col];
            B[row * n + col] = B_s[col];
        }
    }
}

__global__ void kernel_column_norm(int n, DATA_TYPE *X, const DATA_TYPE *B)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n)
    {
        int last_col_idx = row * n + (n - 1);
        if (B[last_col_idx] != 0.f)
            X[last_col_idx] /= B[last_col_idx];
    }
}

__global__ void kernel_column_back_sostitution_shared(int n, DATA_TYPE *X, const DATA_TYPE *A, DATA_TYPE *B)
{
    __shared__ DATA_TYPE X_s[BLOCK_SIZE];
    __shared__ DATA_TYPE B_s[BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = threadIdx.x;

    if (row < n)
    {
        if (col < n)
        {
            X_s[col] = X[row * n + col];
            B_s[col] = B[row * n + col];
        }
        __syncthreads();

        for (int k = n - 2; k >= 0; k--)
        {
            if (col == k && B_s[k] != 0.f)
            {
                X_s[k] = (X_s[k] - X_s[k + 1] * A[row * n + k + 1]) / B_s[k];
            }
            __syncthreads();
        }

        if (col < n)
        {
            X[row * n + col] = X_s[col];
        }
    }
}

__global__ void kernel_row_forward_elimination_shared(int n, DATA_TYPE *X, const DATA_TYPE *A, DATA_TYPE *B)
{
    __shared__ DATA_TYPE X_s[BLOCK_SIZE];
    __shared__ DATA_TYPE B_s[BLOCK_SIZE];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = threadIdx.y;

    if (col < n)
    {
        if (row < n)
        {
            X_s[row] = X[row * n + col];
            B_s[row] = B[row * n + col];
        }
        __syncthreads();

        for (int k = 1; k < n; k++)
        {
            if (row == k && B_s[k - 1] != 0.f)
            {
                X_s[k] -= X_s[k - 1] * A[k * n + col] / B_s[k - 1];
                B_s[k] -= A[k * n + col] * A[k * n + col] / B_s[k - 1];
            }
            __syncthreads();
        }

        if (row < n)
        {
            X[row * n + col] = X_s[row];
            B[row * n + col] = B_s[row];
        }
    }
}

int main()
{
    const int n = N;
    const int tsteps = TSTEPS;
    const int bytes = sizeof(DATA_TYPE) * n * n;

    DATA_TYPE *X, *B;
    gpuErrchk(cudaMallocManaged(&X, bytes));
    gpuErrchk(cudaMallocManaged(&B, bytes));
    DATA_TYPE *A_dev;
    gpuErrchk(cudaMalloc(&A_dev, bytes));

    DATA_TYPE *A_host = (DATA_TYPE *)malloc(bytes);
    DATA_TYPE *X_copy = (DATA_TYPE *)malloc(bytes);
    DATA_TYPE *B_copy = (DATA_TYPE *)malloc(bytes);

    // Inizializzazione
    #pragma omp parallel
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int idx = i * n + j;
            X[idx] = ((DATA_TYPE)i * (j + 1) + 1) / n;
            B[idx] = ((DATA_TYPE)i * (j + 3) + 3) / n;
            A_host[idx] = ((DATA_TYPE)i * (j + 2) + 2) / n;
            X_copy[idx] = X[idx];
            B_copy[idx] = B[idx];
        }
    }

    gpuErrchk(cudaMemcpy(A_dev, A_host, bytes, cudaMemcpyHostToDevice));

    // Call host ADI
    printf("Esecuzione Host...\n");
    struct timespec rt[2];
    clock_gettime(CLOCK_REALTIME, &rt[0]);
    host_adi(tsteps, n, X_copy, A_host, B_copy);
    clock_gettime(CLOCK_REALTIME, &rt[1]);
    double host_time = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("Host ADI completato in: %9.3f sec\n", host_time);

    // Call GPU ADI
    printf("Esecuzione GPU...\n");
    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    clock_gettime(CLOCK_REALTIME, &rt[0]);

    for (int t = 0; t < tsteps; t++)
    {
        // Update along columns
        kernel_column_forward_elimination_shared<<<grid, block>>>(n, X, A_dev, B);
        gpuErrchk(cudaDeviceSynchronize());

        kernel_column_norm<<<grid, block>>>(n, X, B);
        gpuErrchk(cudaDeviceSynchronize());

        kernel_column_back_sostitution_shared<<<grid, block>>>(n, X, A_dev, B);
        gpuErrchk(cudaDeviceSynchronize());

        // Update along rows
        kernel_row_forward_elimination_shared<<<grid, block>>>(n, X, A_dev, B);
        gpuErrchk(cudaDeviceSynchronize());

        kernel_column_norm<<<grid, block>>>(n, X, B);
        gpuErrchk(cudaDeviceSynchronize());

        kernel_column_back_sostitution_shared<<<grid, block>>>(n, X, A_dev, B);
        gpuErrchk(cudaDeviceSynchronize());
    }

    clock_gettime(CLOCK_REALTIME, &rt[1]);
    double gpu_time = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("GPU ADI completato in: %9.3f sec\n", gpu_time);

    // Confronto tra i risultati
    if (compare_matrices(X, X_copy, n))
    {
        printf("Risultati CORRETTI!\n");
    }
    else
    {
        printf("Errore: i risultati non corrispondono.\n");
    }

    // Cleanup
    free(X_copy);
    free(B_copy);
    free(A_host);
    gpuErrchk(cudaFree(X));
    gpuErrchk(cudaFree(B));
    gpuErrchk(cudaFree(A_dev));

    return 0;
}

