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
	#pragma omp parallel
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			X[i * n + j] 			= ((DATA_TYPE)i * (j + 1) + 1) / n;
			X_DEV[i * n + j] 	= ((DATA_TYPE)i * (j + 1) + 1) / n;
			A[i * n + j] 			= ((DATA_TYPE)i * (j + 2) + 2) / n;
			B[i * n + j] 			= ((DATA_TYPE)i * (j + 3) + 3) / n;
			B_DEV[i * n + j] 	= ((DATA_TYPE)i * (j + 3) + 3) / n;
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

// Kernel per aggiornamento lungo le colonne
__global__ void adi_column_update(int n, DATA_TYPE *X, DATA_TYPE *A, DATA_TYPE *B)
{
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  if (i1 < n && i2 > 0 && i2 < n)
  {
    X[i1 * n + i2] -= X[i1 * n + i2 - 1] * A[i1 * n + i2] / B[i1 * n + i2 - 1];
    B[i1 * n + i2] -= A[i1 * n + i2] * A[i1 * n + i2] / B[i1 * n + i2 - 1];
  }
}

// Kernel per normalizzazione lungo l'ultima colonna
__global__ void adi_column_normalize(int n, DATA_TYPE *X, DATA_TYPE *B)
{
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;

  if (i1 < n)
    X[i1 * n + (n - 1)] /= B[i1 * n + (n - 1)];
  
}

// Kernel per back-substitution lungo le colonne
__global__ void adi_column_backsub(int n, DATA_TYPE *X, DATA_TYPE *A, DATA_TYPE *B)
{
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;

  if (i1 < n && i2 < n - 2)
  {
    X[i1 * n + (n - i2 - 2)] = (X[i1 * n + (n - i2 - 2)] - 
                                X[i1 * n + (n - i2 - 3)] * A[i1 * n + (n - i2 - 3)]) / 
                                B[i1 * n + (n - i2 - 3)];
  }
}

// Kernel per aggiornamento lungo le righe
__global__ void adi_row_update(int n, DATA_TYPE *X, DATA_TYPE *A, DATA_TYPE *B)
{
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;

  if (i1 > 0 && i1 < n && i2 < n)
  {
    X[i1 * n + i2] -= X[(i1 - 1) * n + i2] * A[i1 * n + i2] / B[(i1 - 1) * n + i2];
    B[i1 * n + i2] -= A[i1 * n + i2] * A[i1 * n + i2] / B[(i1 - 1) * n + i2];
  }
}

// Kernel per normalizzazione lungo l'ultima riga
__global__ void adi_row_normalize(int n, DATA_TYPE *X, DATA_TYPE *B)
{
  int i2 = blockIdx.x * blockDim.x + threadIdx.x;

  if (i2 < n)
    X[(n - 1) * n + i2] /= B[(n - 1) * n + i2];
  
}

// Kernel per back-substitution lungo le righe
__global__ void adi_row_backsub(int n, DATA_TYPE *X, DATA_TYPE *A, DATA_TYPE *B)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < n - 2 && row < n)
  {
    X[(n - 2 - col) * n + row] = (X[(n - 2 - col) * n + row] - 
                                  X[(n - col - 3) * n + row] * A[(n - 3 - col) * n + row]) / 
                                  B[(n - 2 - col) * n + row];
  }
}

// Funzione principale per eseguire il metodo ADI su GPU
void kernel_adi_device(int tsteps, int n, DATA_TYPE *d_X, DATA_TYPE *d_A, DATA_TYPE *d_B)
{
  const dim3 block(
    BLOCK_SIZE, 
    BLOCK_SIZE
  );
  const dim3 grid1D(
    (n + BLOCK_SIZE - 1) / BLOCK_SIZE
  );
  const dim3 grid2D(
    (n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
    (n + BLOCK_SIZE - 1) / BLOCK_SIZE
  );

	for (int t = 0; t < tsteps; t++)
	{
		// Aggiornamento lungo le colonne
		adi_column_update<<<grid2D, block>>>(n, d_X, d_A, d_B);
		cudaDeviceSynchronize();

		// Normalizzazione lungo l'ultima colonna
		adi_column_normalize<<<grid1D, block>>>(n, d_X, d_B);
		cudaDeviceSynchronize();

		// Back-substitution lungo le colonne
		adi_column_backsub<<<grid2D, block>>>(n, d_X, d_A, d_B);
		cudaDeviceSynchronize();

		// Aggiornamento lungo le righe
		adi_row_update<<<grid2D, block>>>(n, d_X, d_A, d_B);
		cudaDeviceSynchronize();

		// Normalizzazione lungo l'ultima riga
		adi_row_normalize<<<grid1D, block>>>(n, d_X, d_B);
		cudaDeviceSynchronize();

		// Back-substitution lungo le righe
		adi_row_backsub<<<grid2D, block>>>(n, d_X, d_A, d_B);
		cudaDeviceSynchronize();
	}
}


__global__  void kernel_update_columns(int tsteps, int n, DATA_TYPE *d_X, DATA_TYPE *d_A, DATA_TYPE *d_B)
{
  
}
__global__  void kernel_update_rows(int tsteps, int n, DATA_TYPE *d_X, DATA_TYPE *d_A, DATA_TYPE *d_B)
{
  
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
	const int n = N;
	const int tsteps = TSTEPS;
	const int bytes = sizeof(DATA_TYPE) * n * n;
	struct timespec rt[2];

	DATA_TYPE* X 			= (DATA_TYPE*)malloc(bytes);
	DATA_TYPE* X_dev 	= (DATA_TYPE*)malloc(bytes);
	DATA_TYPE* A 			= (DATA_TYPE*)malloc(bytes);
	DATA_TYPE* B 			= (DATA_TYPE*)malloc(bytes);
	DATA_TYPE* B_dev 	= (DATA_TYPE*)malloc(bytes);
	init_array(n, X, X_dev, A, B, B_dev);

	// Kernel host
	{
		clock_gettime(CLOCK_REALTIME, rt);
		kernel_adi_host(tsteps, n, X, A, B);
		clock_gettime(CLOCK_REALTIME, rt + 1);

		double wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
		printf("ADI (Host): %9.3f sec\n", wt);
	}

	// Allocazione memoria GPU
	DATA_TYPE *d_X, *d_A, *d_B;
	cudaMalloc(&d_X, bytes);
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);
	cudaMemcpy(d_X, X_dev, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B_dev, bytes, cudaMemcpyHostToDevice);

	// Dimensionamento Griglia e Blocco (16 - con 32 non parte)
	{
    const dim3 block(
      BLOCK_SIZE, 
      BLOCK_SIZE
    );
    const dim3 grid2D(
      (n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
      (n + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

		clock_gettime(CLOCK_REALTIME, rt);

    for (int t = 0; t < tsteps; t++)
    {
      //The ADI method is a two step iteration process that alternately updates the column and row spaces 
      // of an approximate solution to AX - XB = C. 
      // One ADI iteration consists of the following steps:
      // 1. Solve for X^(j+1/2), where 
      //   (A - β_(j+1)I)X^(j+1/2) = X^(j)(B - β_(j+1)I) + C.
      // 2. Solve for X^(j+1), where 
      //   X^(j+1)(B - α_(j+1)I) = (A - α_(j+1)I)X^(j+1/2) - C.

      kernel_update_columns<<<grid, block>>>(n, d_X, d_A, d_B);
      cudaDeviceSynchronize();  // Sincronizzazione per sicurezza

      kernel_update_rows<<<grid, block>>>(n, d_X, d_A, d_B);
      cudaDeviceSynchronize();
    }



		//kernel_adi_device(tsteps, n, d_X, d_A, d_B);		
    
    
    
    cudaMemcpy(X_dev, d_X, bytes, cudaMemcpyDeviceToHost);
		gpuErrchk(cudaPeekAtLastError());  
	
		clock_gettime(CLOCK_REALTIME, rt + 1);

		double wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
		printf("ADI (GPU): %9.3f sec\n", wt);
	}


	if (compare_matrices(X, X_dev, n))
	{
		printf("Risultati Host e Device CORRETTI!\n");
	}
	else
	{
		printf("Risultati Host e Device NON corrispondono!\n");
	}

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
