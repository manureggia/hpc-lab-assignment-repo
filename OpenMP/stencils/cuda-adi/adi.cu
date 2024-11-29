/* POLYBENCH/GPU-OPENMP
 *
 * This file is a part of the Polybench/GPU-OpenMP suite
 *
 * Contact:
 * William Killian <killian@udel.edu>
 *
 * Copyright 2013, The University of Delaware
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 10x1024x1024. */
#include "adi.h"

#if !defined(NTHREADS_GPU)
#define NTHREADS_GPU (1024)
#endif

#define NO_OPT 0
#define HOST 1
#define DEVICE 2

#ifndef BLOCK_SIZE
#define BLOCK_SIZE (512)
#endif

#if !defined(OPT_TYPE)
#define OPT_TYPE HOST
#endif

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

/* Array initialization. */
static void init_array(int n,
                       DATA_TYPE POLYBENCH_2D(X, N, N, n, n),
                       DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                       DATA_TYPE POLYBENCH_2D(B, N, N, n, n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
      X[i][j] = ((DATA_TYPE)i * (j + 1) + 1) / n;
      A[i][j] = ((DATA_TYPE)i * (j + 2) + 2) / n;
      B[i][j] = ((DATA_TYPE)i * (j + 3) + 3) / n;
    }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n,
                        DATA_TYPE POLYBENCH_2D(X, N, N, n, n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, X[i][j]);
      if ((i * N + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


static void kernel_adi_host(int tsteps, int n, DATA_TYPE POLYBENCH_2D(X, N, N, n, n), DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_2D(B, N, N, n, n))
{
  for (int t = 0; t < _PB_TSTEPS; t++)
  {
    for (int i1 = 0; i1 < _PB_N; i1++)
    {
      for (int i2 = 1; i2 < _PB_N; i2++)
      {
        X[i1][i2] = X[i1][i2] - X[i1][i2 - 1] * A[i1][i2] / B[i1][i2 - 1];
        B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2 - 1];
      }
    }


    for (int i1 = 0; i1 < _PB_N; i1++)
      X[i1][_PB_N - 1] = X[i1][_PB_N - 1] / B[i1][_PB_N - 1];


    for (int i1 = 0; i1 < _PB_N; i1++)
    {
      for (int i2 = 0; i2 < _PB_N - 2; i2++)
        X[i1][_PB_N - i2 - 2] = (X[i1][_PB_N - 2 - i2] - X[i1][_PB_N - 2 - i2 - 1] * A[i1][_PB_N - i2 - 3]) / B[i1][_PB_N - 3 - i2];
    }

    for (int i1 = 1; i1 < _PB_N; i1++)
    {
      for (int i2 = 0; i2 < _PB_N; i2++)
      {
        X[i1][i2] = X[i1][i2] - X[i1 - 1][i2] * A[i1][i2] / B[i1 - 1][i2];
        B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1 - 1][i2];
      }
    }

    for (int i2 = 0; i2 < _PB_N; i2++)
      X[_PB_N - 1][i2] = X[_PB_N - 1][i2] / B[_PB_N - 1][i2];

    for (int i1 = 0; i1 < _PB_N - 2; i1++)
    {
      for (int i2 = 0; i2 < _PB_N; i2++)
        X[_PB_N - 2 - i1][i2] = (X[_PB_N - 2 - i1][i2] - X[_PB_N - i1 - 3][i2] * A[_PB_N - 3 - i1][i2]) / B[_PB_N - 2 - i1][i2];
    }
  }
}

static void kernel_adi_host_parallel(
  int tsteps,
  int n,
  DATA_TYPE POLYBENCH_2D(X, N, N, n, n),
  DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
  DATA_TYPE POLYBENCH_2D(B, N, N, n, n)
)
{
  for (int t = 0; t < _PB_TSTEPS; t++)
  {
    #pragma omp parallel
    {
      /**
       * Updating of X and B along the columns:
       * This loop modifies X and B based on previous values in the same row by iterating over 
       * columns (i2). This type of update is similar to a forward deletion that normalizes 
       * X and B against the previous elements.
       */
      #pragma omp for collapse(2)
      for (int i1 = 0; i1 < _PB_N; i1++)
      {
        for (int i2 = 1; i2 < _PB_N; i2++)
        {
          X[i1][i2] = X[i1][i2] - X[i1][i2 - 1] * A[i1][i2] / B[i1][i2 - 1];
          B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2 - 1];
        }
      }
      
      /**
       * Normalization of the last element of the row in X:
       * This loop divides the last element of each row in X by the corresponding element in B, 
       * completing the normalization process for that row.
       */
      #pragma omp for simd
      for (int i1 = 0; i1 < _PB_N; i1++)
        X[i1][_PB_N - 1] = X[i1][_PB_N - 1] / B[i1][_PB_N - 1];

      /**
       * Back-substitution along the columns: 
       * This cycle performs a back-substitution on X, starting from the last element and moving 
       * toward the beginning of the row. It is similar to a back-substitution step to solve 
       * a triangular system.
       */
      #pragma omp for collapse(2)
      for (int i1 = 0; i1 < _PB_N; i1++)
      {
        for (int i2 = 0; i2 < _PB_N - 2; i2++)
          X[i1][_PB_N - i2 - 2] = (X[i1][_PB_N - 2 - i2] - X[i1][_PB_N - 2 - i2 - 1] * A[i1][_PB_N - i2 - 3]) / B[i1][_PB_N - 3 - i2];
      }

      /**
       * Updating X and B along the rows: 
       * This loop does a normal “forward elimination” along the rows, 
       * similar to the first step but in the vertical direction, updating X and B 
       * according to the values in the previous rows.
       */
      
      for (int i1 = 1; i1 < _PB_N; i1++)
      {
        #pragma omp for
        for (int i2 = 0; i2 < _PB_N; i2++)
        {
          X[i1][i2] = X[i1][i2] - X[i1 - 1][i2] * A[i1][i2] / B[i1 - 1][i2];
          B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1 - 1][i2];
        }
      }

      /**
       * Normalization of the last column element in X: This loop divides the last element of 
       * each column of X by the corresponding value of B, completing the normalization of the 
       * column.
       */
      #pragma omp for simd
      for (int i2 = 0; i2 < _PB_N; i2++)
        X[_PB_N - 1][i2] = X[_PB_N - 1][i2] / B[_PB_N - 1][i2];

      /**
       * Back-substitution along rows: Here another back-substitution is performed, 
       * this time along rows, to resolve each element of X according to B and A.
       */
      for (int i1 = 0; i1 < _PB_N - 2; i1++)
      {
        #pragma omp for 
        for (int i2 = 0; i2 < _PB_N; i2++)
          X[_PB_N - 2 - i1][i2] = (X[_PB_N - 2 - i1][i2] - X[_PB_N - i1 - 3][i2] * A[_PB_N - 3 - i1][i2]) / B[_PB_N - 2 - i1][i2];
      }
    }
  }
}

/**
 * Main computational kernel. The whole function will be timed, including the call and return.
 * 
 * The kernel_adi code is a computational kernel that applies a series of transformations 
 * to the matrix X using the matrices A and B, following the Alternating Direction Implicit 
 * (ADI) methodology. This technique is often used to solve partial differential equations 
 * (PDEs), especially in diffusion and transport problems.
*/
#elif OPT_TYPE == DEVICE
static void kernel_adi(int tsteps, int n, DATA_TYPE POLYBENCH_2D(X, N, N, n, n),
                      DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                      DATA_TYPE POLYBENCH_2D(B, N, N, n, n))
{
    #pragma omp target data map(tofrom: X[0:n][0:n]) map(to: A[0:n][0:n], B[0:n][0:n])
    #pragma omp target parallel
      for (int t = 0; t < _PB_TSTEPS; t++)
      {
          // First phase: Column sweeps
          #pragma omp for collapse(2)
          for (int i1 = 0; i1 < _PB_N; i1++)
          {
              for (int i2 = 1; i2 < _PB_N; i2++)
              {
                  X[i1][i2] = X[i1][i2] - X[i1][i2-1] * A[i1][i2] / B[i1][i2-1];
                  B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2-1];
              }
          }
          
          #pragma omp barrier

          // Update last column
          #pragma omp for
          for (int i1 = 0; i1 < _PB_N; i1++)
              X[i1][_PB_N-1] = X[i1][_PB_N-1] / B[i1][_PB_N-1];
          
          #pragma omp barrier

          // Backward sweep
          #pragma omp for collapse(2)
          for (int i1 = 0; i1 < _PB_N; i1++)
          {
              for (int i2 = 0; i2 < _PB_N-2; i2++)
              {
                  X[i1][_PB_N-i2-2] = (X[i1][_PB_N-2-i2] - 
                                      X[i1][_PB_N-2-i2-1] * A[i1][_PB_N-i2-3]) / 
                                      B[i1][_PB_N-3-i2];
              }
          }
          
          #pragma omp barrier

          // Second phase: Row sweeps
          #pragma omp for collapse(2)
          for (int i1 = 1; i1 < _PB_N; i1++)
          {
              for (int i2 = 0; i2 < _PB_N; i2++)
              {
                  X[i1][i2] = X[i1][i2] - X[i1-1][i2] * A[i1][i2] / B[i1-1][i2];
                  B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1-1][i2];
              }
          }
          
          #pragma omp barrier

          // Update last row
          #pragma omp for
          for (int i2 = 0; i2 < _PB_N; i2++)
              X[_PB_N-1][i2] = X[_PB_N-1][i2] / B[_PB_N-1][i2];
          
          #pragma omp barrier

          // Backward sweep - Sequential in i1 due to dependencies
          for (int i1 = 0; i1 < _PB_N-2; i1++)
          {
              #pragma omp for
              for (int i2 = 0; i2 < _PB_N; i2++)
              {
                  X[_PB_N-2-i1][i2] = (X[_PB_N-2-i1][i2] - 
                                      X[_PB_N-i1-3][i2] * A[_PB_N-3-i1][i2]) / 
                                      B[_PB_N-2-i1][i2];
              }
              #pragma omp barrier
          }
      }
  }

#endif

__global__ void kernel_adi(int tsteps, int n, DATA_TYPE *X, DATA_TYPE *A, DATA_TYPE *B) {
    int i1 = blockIdx.x * blockDim.x + threadIdx.x;
    int i2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i1 < n && i2 < n) {
        for (int t = 0; t < tsteps; t++) {
            if (i2 > 0) {
                X[i1 * n + i2] -= X[i1 * n + i2 - 1] * A[i1 * n + i2] / B[i1 * n + i2 - 1];
                B[i1 * n + i2] -= A[i1 * n + i2] * A[i1 * n + i2] / B[i1 * n + i2 - 1];
            }

            __syncthreads();

            if (i2 == n - 1) {
                X[i1 * n + i2] /= B[i1 * n + i2];
            }

            __syncthreads();

            if (i2 < n - 2) {
                X[i1 * n + (n - i2 - 2)] =
                    (X[i1 * n + (n - i2 - 2)] - X[i1 * n + (n - i2 - 3)] * A[i1 * n + (n - i2 - 3)]) /
                    B[i1 * n + (n - i2 - 3)];
            }

            __syncthreads();

            if (i1 > 0) {
                X[i1 * n + i2] -= X[(i1 - 1) * n + i2] * A[i1 * n + i2] / B[(i1 - 1) * n + i2];
                B[i1 * n + i2] -= A[i1 * n + i2] * A[i1 * n + i2] / B[(i1 - 1) * n + i2];
            }

            __syncthreads();

            if (i1 == n - 1) {
                X[i1 * n + i2] /= B[i1 * n + i2];
            }

            __syncthreads();

            if (i1 < n - 2) {
                X[(n - i1 - 2) * n + i2] =
                    (X[(n - i1 - 2) * n + i2] - X[(n - i1 - 3) * n + i2] * A[(n - i1 - 3) * n + i2]) /
                    B[(n - i1 - 2) * n + i2];
            }

            __syncthreads();
        }
    }
}


int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(X, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(X), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));


  clock_gettime(CLOCK_REALTIME, rt + 0);
  kernel_adi(tsteps, n, POLYBENCH_ARRAY(X), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));
  clock_gettime(CLOCK_REALTIME, rt + 1);
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf("ADI (Host) : %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * n * n * n / (1.0e9 * wt));

  /* start second timer */
  clock_gettime(CLOCK_REALTIME, rt + 0);
  /*cuda mem copy */

  gpuErrchk(cudaMemcpy(d_X, X, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_A, A, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_B, B, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice));
  
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((n + (BLOCK_SIZE)-1) / (BLOCK_SIZE), (n + (BLOCK_SIZE)-1) / (BLOCK_SIZE));

  kernel_adi<<<gridDim, blockDim>>>(tsteps, n, d_X, d_A, d_B);

  clock_gettime(CLOCK_REALTIME, rt + 1);
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf("ADI-v1 (GPU): %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * n * n * n / (1.0e9 * wt));


  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(X)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(X);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
