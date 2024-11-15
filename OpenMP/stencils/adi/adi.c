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

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 10x1024x1024. */
#include "adi.h"

#if !defined(NTHREADS_GPU)
#define NTHREADS_GPU (1024)
#endif

#define HOST 1
#define DEVICE 2

#if !defined(OPT_TYPE)
#define OPT_TYPE HOST
#endif

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

#if OPT_TYPE == HOST
static void kernel_adi(
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
      #pragma omp for
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
      #pragma omp for
      for (int i1 = 0; i1 < _PB_N; i1++)
        X[i1][_PB_N - 1] = X[i1][_PB_N - 1] / B[i1][_PB_N - 1];

      /**
       * Back-substitution along the columns: 
       * This cycle performs a back-substitution on X, starting from the last element and moving 
       * toward the beginning of the row. It is similar to a back-substitution step to solve 
       * a triangular system.
       */
      #pragma omp for
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
      #pragma omp for
      for (int i1 = 1; i1 < _PB_N; i1++)
      {
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
      #pragma omp for
      for (int i2 = 0; i2 < _PB_N; i2++)
        X[_PB_N - 1][i2] = X[_PB_N - 1][i2] / B[_PB_N - 1][i2];

      /**
       * Back-substitution along rows: Here another back-substitution is performed, 
       * this time along rows, to resolve each element of X according to B and A.
       */
      #pragma omp for
      for (int i1 = 0; i1 < _PB_N - 2; i1++)
      {
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
static void kernel_adi(
  int tsteps,
  int n,
  DATA_TYPE POLYBENCH_2D(X, N, N, n, n),
  DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
  DATA_TYPE POLYBENCH_2D(B, N, N, n, n)
)
{
	#pragma omp target data map(to:A[0:_PB_N][0:_PB_N]) map(tofrom:X[0:_PB_N][0:_PB_N], B[0:_PB_N][0:_PB_N])
  for (int t = 0; t < _PB_TSTEPS; t++)
  {
		/**
			* Updating of X and B along the columns:
			* This loop modifies X and B based on previous values in the same row by iterating over 
			* columns (i2). This type of update is similar to a forward deletion that normalizes 
			* X and B against the previous elements.
			*/
		#pragma omp target teams distribute parallel for collapse(2)
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
		#pragma omp target teams distribute parallel for
		for (int i1 = 0; i1 < _PB_N; i1++)
			X[i1][_PB_N - 1] = X[i1][_PB_N - 1] / B[i1][_PB_N - 1];

		/**
			* Back-substitution along the columns: 
			* This cycle performs a back-substitution on X, starting from the last element and moving 
			* toward the beginning of the row. It is similar to a back-substitution step to solve 
			* a triangular system.
			*/
		#pragma omp target teams distribute parallel for collapse(2)
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
		#pragma omp target teams distribute parallel for collapse(2)
		for (int i1 = 1; i1 < _PB_N; i1++)
		{
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
		#pragma omp target teams distribute parallel for
		for (int i2 = 0; i2 < _PB_N; i2++)
			X[_PB_N - 1][i2] = X[_PB_N - 1][i2] / B[_PB_N - 1][i2];

		/**
			* Back-substitution along rows: Here another back-substitution is performed, 
			* this time along rows, to resolve each element of X according to B and A.
			*/
		#pragma omp target teams distribute parallel for collapse(2)
		for (int i1 = 0; i1 < _PB_N - 2; i1++)
		{
			for (int i2 = 0; i2 < _PB_N; i2++)
				X[_PB_N - 2 - i1][i2] = (X[_PB_N - 2 - i1][i2] - X[_PB_N - i1 - 3][i2] * A[_PB_N - 3 - i1][i2]) / B[_PB_N - 2 - i1][i2];
		}
  }
}
#endif

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

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_adi(tsteps, n, POLYBENCH_ARRAY(X),
             POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;



  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(X)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(X);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
