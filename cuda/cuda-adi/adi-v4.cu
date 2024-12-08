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
			printf("%0.2f", X[i * n + j]);
			if ((i * n + j) % n == 0)
				printf("\n");
		}
	}
	printf("\n");
}

// Confronta due matrici per verificare la correttezza
int compare_matrices(DATA_TYPE *X, DATA_TYPE *X_copy, int n)
{
  int return_value = EXIT_SUCCESS;
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
			DATA_TYPE x1 = (DATA_TYPE) X[i * n + j];
			DATA_TYPE x2 = (DATA_TYPE) X_copy[i * n + j];
			DATA_TYPE diff = fabs(x1 - x2);

			if(isnan(x1) || isnan(x2))
				printf("Comparing with nan\n");

      if (diff > 1e-6)
      {
        printf("Mismatch at (%d, %d): Device = %f, Host = %f\n", i, j, x1, x2);
        return_value = EXIT_FAILURE;
      }
    }
  }
  return return_value; 
}

void inline host_adi_col_forward_elimination(int n, DATA_TYPE *X, const DATA_TYPE *A, DATA_TYPE *B)
{
	for (int row = 0; row < n; row++)
	{
		for (int col = 1; col < n; col++)
		{
			int idx = row * n + col;
			int prev_idx = row * n + (col - 1);
			if(B[prev_idx] != 0.f)
			{
				X[idx] -= X[prev_idx] * A[idx] / B[prev_idx];
				B[idx] -= A[idx] * A[idx] / B[prev_idx];
			}
		}
	}
}
void inline host_adi_col_norm(int n, DATA_TYPE *X, const DATA_TYPE *B)
{
	for (int col = 0; col < n; col++)
	{
		int idx = col * n + (n - 1);
		if(B[idx] != 0.f)
			X[idx] /= B[idx];
	}
}
void inline host_adi_col_back_sostitution(int n, DATA_TYPE *X, const DATA_TYPE *A, const DATA_TYPE *B)
{
	for (int row = 0; row < n; row++)
	{
		for (int col = n - 2; col >= 0; col--)
		{
			int idx = row * n + col;
			int next_idx = row * n + (col + 1);
			if (B[idx] != 0.f)
				X[idx] = (X[idx] - X[next_idx] * A[next_idx]) / B[idx];
		}
	}
}
void inline host_adi_row_forward_elimination(int n, DATA_TYPE *X, const DATA_TYPE *A, DATA_TYPE *B)
{
	for (int row = 1; row < n; row++)
	{
		for (int col = 0; col < n; col++)
		{
			int idx = row * n + col;
			int prev_idx = (row - 1) * n + col;
			if (B[prev_idx] != 0.f)
			{
				X[idx] -= X[prev_idx] * A[idx] / B[prev_idx];
				B[idx] -= A[idx] * A[idx] / B[prev_idx];
			}
		}
	}
}
void inline host_adi_row_norm(int n, DATA_TYPE *X, const DATA_TYPE *B)
{
	for (int col = 0; col < n; col++)
	{
		int idx = (n - 1) * n + col;
		if(B[idx] != 0.f)
			X[idx] /= B[idx];
	}
}
void inline host_adi_row_back_sostitution(int n, DATA_TYPE *X, const DATA_TYPE *A, DATA_TYPE *B)
{
	for (int row = n - 2; row >= 0; row--) 
	{
		for (int col = 0; col < n; col++)
		{
			int idx = row * n + col;
			int next_idx = (row + 1) * n + col;
			if(B[idx] != 0.f)
				X[idx] = (X[idx] - X[next_idx] * A[next_idx]) / B[idx];
		}
	}
}
void inline host_adi(int tsteps, int n, DATA_TYPE *X, const DATA_TYPE *A, DATA_TYPE *B)
{
	/** At each iteration, the system is updated along the columns and rows */
	for (int t = 0; t < tsteps; t++)
	{
		// -----------------------------------
		// Update along the columns
		// -----------------------------------
		// Forward elimination is performed
		host_adi_col_forward_elimination(n, X, A, B);
		// Normalization
		host_adi_col_norm(n, X, B);
		// Back Substitution is performed
		host_adi_col_back_sostitution(n, X, A, B);

		// -----------------------------------
		// Update along the rows
		// -----------------------------------
		// Forward elimination is performed
		host_adi_row_forward_elimination(n, X, A, B);
		// Normalization
		host_adi_row_norm(n, X, B);
		// Back Substitution is performed
		host_adi_row_back_sostitution(n, X, A, B);
	}
}

__global__ void kernel_column_forward_elimination(int n, DATA_TYPE *X, const DATA_TYPE *A, DATA_TYPE *B)
{
	/**
	 * 1. Each block is one-dimensional along the y-axis. 
	 * 		Thus, there are BLOCK_SIZE threads organized vertically per block.
	 * 2. The row on which each thread operates is calculated as: row = blockIdx.y * blockDim.y + threadIdx.y;
	 * 	 
	 * The elements of X, A, and B are used multiple times within the same thread block, 
	 * they can be loaded into shared memory.
	 * Using shared memory CAN reduces the number of global memory accesses, minimizing the latency time.
	 */

	// Version 0: no shared memory
	// ------------------------------------------------------
#if 1
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < n)
	{
		// For each row, the thread iterates over all columns from left to right
		for (int col = 1; col < n; col++)
		{
			// Updates for X and B are made based on the previous values in the same row.
			// This depends on the structure of the problem, which has a directional dependence along the columns.
			int idx = row * n + col;
			int prev_idx = row * n + (col - 1);
			if(B[prev_idx] != 0.f)
			{
				X[idx] -= X[prev_idx] * A[idx] / B[prev_idx];
				B[idx] -= A[idx] * A[idx] / B[prev_idx];
			}
		}
	}
#endif

	// Version 1: X and B in shared memory 
	// ------------------------------------------------------
#if 0
	__shared__ DATA_TYPE shared_X[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ DATA_TYPE shared_B[BLOCK_SIZE][BLOCK_SIZE];
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < n)
	{
		// Since it is not possible to copy an entire row to the shared memory
		// because it is too large, I have to load the rows a little at a time in chunks.
		// The chunk_size will be equal to BLOCK_SIZE; each iteration loads a chunk of the row in X
		// on the shared memory.
		// The number of chunks needed to load the entire row will be equal to ((n + chunk_size - 1) / chunk_size)
		int chunk_size = BLOCK_SIZE;
		int num_chunks = (n + chunk_size - 1) / chunk_size;
		for(int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++)
		{
			// 1. Carica i dati di questo chunk nella memoria condivisa
			int chunk_start = chunk_idx * chunk_size;
			int chunk_end = min(chunk_start + chunk_size, n);
			int local_row = threadIdx.y;
			for (int col = chunk_start; col < chunk_end; col++)
			{
				int local_col = col - chunk_start;
				shared_X[local_row][local_col] = X[row * n + col];
				shared_B[local_row][local_col] = B[row * n + col];
			}
	
			// 2. Updating values in Shared Memory
			for (int col = 1; col < chunk_size; col++)
			{
				int global_col = chunk_start + col;
				if (global_col < n)
				{
					DATA_TYPE a = A[row * n + global_col];
					DATA_TYPE b = shared_B[local_row][col - 1];
					if(b != 0.f)
					{
						shared_X[local_row][col] -= shared_X[local_row][col - 1] * a / shared_B[local_row][col - 1];
						shared_B[local_row][col] -= a * a / ;
					}
				}
			}

			// 3. Writing in global memory
			for (int col = chunk_start; col < chunk_end; col++)
			{
				int local_col = col - chunk_start;
				X[row * n + col] = shared_X[local_row][local_col];
				B[row * n + col] = shared_B[local_row][local_col];
			}
		}
	}
#endif

	// Version 2: read-only matrix A in shared memory
	// ------------------------------------------------------
#if 0
	__shared__ DATA_TYPE shared_A[BLOCK_SIZE][BLOCK_SIZE];
	int global_row = blockIdx.y * blockDim.y + threadIdx.y;
	if (global_row < n)
	{
		// Since it is not possible to copy an entire row to the shared memory
		// because it is too large, I have to load the rows a little at a time in chunks.
		// The chunk_size will be equal to BLOCK_SIZE; each iteration loads a chunk of the row in X
		// on the shared memory.
		// The number of chunks needed to load the entire row will be equal to ((n + chunk_size - 1) / chunk_size)
		int chunk_size = BLOCK_SIZE;
		int num_chunks = (n + chunk_size - 1) / chunk_size;
		for(int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++)
		{
			// Calculation of chunk limits
			int chunk_start = chunk_idx * chunk_size;
			int chunk_end = min(chunk_start + chunk_size, n);

			// Loading into shared memory
			int local_row = threadIdx.y;
			for (int global_col = chunk_start; global_col < chunk_end; global_col++)
			{
				int local_col = global_col - chunk_start;
				shared_A[local_row][local_col] = A[global_row * n + global_col];
			}
			__syncthreads();

			// Updating the values of X and B using shared_A
			for (int global_col = chunk_start + 1; global_col < chunk_end; global_col++)
			{
				int local_col = global_col - chunk_start;
				int idx = global_row * n + global_col;
				int prev_idx = global_row * n + (global_col - 1);
				DATA_TYPE a = shared_A[local_row][local_col];
				if(B[prev_idx] != 0.f)
				{
					X[idx] -= X[prev_idx] * a / B[prev_idx];
					B[idx] -= a * a / B[prev_idx];
				}
			}
		}
	}
#endif

	/**
	 * Conclusions: version 2, in which shared memory is used only for the A (read-only) array, 
	 * performs better than version 1 for several reasons:
	 * 1. Reduced use of shared memory; version 1 uses two shared arrays (shared_X and shared_B),
	 * 		thus higher memory consumption per block and this results in:
	 * 		- reduced number of active blocks that can run simultaneously on a single multiprocessor (SM) GPU
	 * 			reducing parallelism
	 * 2. Avoiding excessive synchronization; version 1 requires many more calls to __syncthreads() than version 2.
	 * 3. Martice A is read frequently when updating X and B, so loading it into shared memory reduces 
	 * 		significantly the number of accesses to global memory, which is much slower than shared memory.
	 */
}
__global__ void kernel_column_norm(int n, DATA_TYPE *X, const DATA_TYPE *B)
{
	/**
	 * The kernel kernel_column_norm is intended to normalize the last column 
	 * of the matrix X with respect to B for each row.
	 * 
	 * This kernel acts on a single column (n-1). 
	 * The use of shared memory can be avoided, since no complex iterative calculations are performed on the same row.
	 * It adds latency with the use of __syncthreads() and may degrade performance
	 */
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < n) 
	{
		// Given the row, the index of the last column is calculated
		// The element X[last_col_idx] is normalized by dividing by B[last_col_idx]
		// Each thread normalizes one element in the last column of a row.
		int last_col_idx = row * n + (n - 1);
		if (B[last_col_idx] != 0.f)
			X[last_col_idx] /= B[last_col_idx];
	}
}
__global__ void kernel_column_back_sostitution(int n, DATA_TYPE *X, const DATA_TYPE *A, const DATA_TYPE *B)
{
	/**
	 * The kernel kernel_column_back_substitution implements backward substitution 
	 * along the columns for each row of the X array. 
	 * Each thread processes one row independent of the others.
	 *
	 * You can use shared memory to avoid repeated accesses to global memory. 
	 * For example, you could load a row of X, A and B into shared memory
	 */

	// Version 0: no shared memory
	// ------------------------------------------------------
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < n) 
	{
		// It starts from the penultimate column (n - 2) and proceeds to the left, 
		// since the backward substitution is based on the values calculated in the subsequent columns
		for (int col = n - 2; col >= 0; col--) 
		{
			int idx = row * n + col;
			int next_idx = row * n + (col + 1);
			if (B[idx] != 0.f)
				X[idx] = (X[idx] - X[next_idx] * A[next_idx]) / B[idx];
		}
	}

}
__global__ void kernel_row_forward_elimination(int n, DATA_TYPE *X, const DATA_TYPE *A, DATA_TYPE *B)
{
	/**
	 * The kernel implements forward elimination along the rows, operating on each column of the X matrix. 
	 * The approach taken assigns each thread a column and parallelizes processing along the column axis.
	 * 	
	 * Each thread works on one column independently of the others, 
	 * making the kernel parallelizable along the x-axis. 
	 * Also The use of global memory for X, A, and B for each access may be inefficient.
	 * The use of shared memory could improve kernel efficiency by reducing global memory accesses.
	 */

	// I associate each thread with a column, using blockIdx.x and threadIdx.x
	// I calculate the Column Index as: col = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < n)
	{
		// Forward elimination proceeds from the second row (row = 1). 
		// The current row depends on the values calculated in the previous row (row - 1).
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
	}

	// TODO: implementazione con shared memory
	// ...
}
__global__ void kernel_row_norm(int n, DATA_TYPE *X, const DATA_TYPE *B)
{
	/**
	 * The kernel takes care of the normalization of the last row of the matrix X,
	 * dividing each element of the row by the corresponding element of the matrix B, again on the last row.
	 * 	
	 * In this case, the use of shared memory is not advantageous. 
	 * Since the kernel operates only on a single row of the matrix and each thread processes an independent element.
	 */

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < n)
	{
		// This calculation determines the global index for the col element of the last row of X and B.
		int last_row_idx = (n - 1) * n + col;
		if(B[last_row_idx] != 0.f)
			X[last_row_idx] /= B[last_row_idx];
	}
}
__global__ void kernel_row_back_sostitution(int n, DATA_TYPE *X, const DATA_TYPE *A, const DATA_TYPE *B)
{
	/**
	 * The kernel kernel_row_back_substitution deals with back substitution (back substitution) 
	 * along the rows of an array. 
	 * Each thread processes a specific column of matrix X, iteratively updating the values of X 
	 * going back up from the second-to-last row (n - 2) to the first (row = 0).
	 * 
	 * Each thread processes an independent column of the matrix. 
	 * This ensures parallelization along the x-axis.
	 * 	
	 * The use of shared memory in this kernel can improve performance,
	 * since it reduces repeated access to global memory.
	 */

	// Each thread is responsible for one column of the matrix, determined by blockIdx.x and threadIdx.x
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < n)
	{
		// Backward substitution starts from the second-to-last row (n - 2) and goes up to the first row (row = 0).
		for (int row = n - 2; row >= 0; row--) 
		{
			int idx = row * n + col;
			int next_idx = (row + 1) * n + col;
			if(B[idx] != 0.f)
				X[idx] = (X[idx] - X[next_idx] * A[next_idx]) / B[idx];
		}
	}

	// TODO: implementazione con shared memory
	// ...
}

int main()
{
	const int n = N;
	const int tsteps = 40;
	const int bytes = sizeof(DATA_TYPE) * n * n;
	struct timespec rt[2];

	// The following data are needed on the GPU side:
	// - X[] read/write -> unified memory 
	// - B[] read/write -> unified memory
	// - A[] read only 	-> unified memory

	// NOTE: con A in pinned memory il tempo HOST_ADI è sui 25 secondi
	// mentre se la alloco con malloc o in memoria unificata il tempo
	// si aggira sui 6-7 secondi

	DATA_TYPE *X, *B, *A;
	gpuErrchk(cudaMallocManaged(&X, bytes));
	gpuErrchk(cudaMallocManaged(&B, bytes));
	gpuErrchk(cudaMallocManaged(&A, bytes));
	DATA_TYPE* X_copy = (DATA_TYPE*)malloc(bytes);
	DATA_TYPE* B_copy = (DATA_TYPE*)malloc(bytes);

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			int idx 			= i * n + j;
			X[idx] 				= ((DATA_TYPE)i * (j + 1) + 1) / n;
			B[idx] 				= ((DATA_TYPE)i * (j + 3) + 3) / n;
			A[idx] 				= ((DATA_TYPE)i * (j + 2) + 2) / n;
			X_copy[idx] 	= ((DATA_TYPE)i * (j + 1) + 1) / n;
			B_copy[idx] 	= ((DATA_TYPE)i * (j + 3) + 3) / n;
		}
	}

	// call ADI on host
	{
		clock_gettime(CLOCK_REALTIME, rt);
		host_adi(tsteps, n, X_copy, A, B_copy);
		clock_gettime(CLOCK_REALTIME, rt + 1);

		double wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
		printf("ADI (Host): %9.3f sec\n", wt);
	}

	// call ADI on GPU
	{
    // const dim3 block(BLOCK_SIZE);
		const int nBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
		dim3 blockLayout;
		dim3 gridLayout;

		clock_gettime(CLOCK_REALTIME, rt);
		/**
		 * As we saw in the host-side implementation, the adi algorithm consists of several
		 * basic steps:
		 * [1] Updating along columns: 
		 * 		[1.1] forward elimination (Forward Elimination).
		 * 		[1.2] normalization
		 * 		[1.3] backward substitution (Back Substitution).
		 * [2] Update along rows:
		 * 		[2.1] forward elimination
		 * 		[2.2] normalization
		 * 		[2.3] back substitution
		 * 		 * 
		 * In the ADI algorithm, some operations can be performed in parallel because they do not depend 
		 * directly on the results of other computations for each spatial iteration.
		 * Operations can be parallelized for rows during the update along columns and 
		 * by columns when updating along rows.
		 */
		for (int t = 0; t < tsteps; t++)
		{
			// ------------------------------------------------
			// Update along the columns
			// ------------------------------------------------
			// Forward Elimination: updates along a column of the same row depend on the previous value
			// in the same row, so it is not parallelizable along columns,
			// but the operation for different rows is independent.
			blockLayout = dim3(1, BLOCK_SIZE, 1);
			gridLayout 	= dim3(1, nBlocks, 1);
			kernel_column_forward_elimination<<<gridLayout, blockLayout>>>(n, X, A, B);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
			// Normalization: parallelizable per row; each row is independent.
			kernel_column_norm<<<gridLayout, blockLayout>>>(n, X, B);
			gpuErrchk(cudaPeekAtLastError());  
			gpuErrchk(cudaDeviceSynchronize());
			// Back Substitution: parallelizable per row; again, each row represents an independent tridiagonal system.
			// The operation along columns depends on the previous values of the same row.
			kernel_column_back_sostitution<<<gridLayout, blockLayout>>>(n, X, A, B);
			gpuErrchk(cudaPeekAtLastError());  
			gpuErrchk(cudaDeviceSynchronize());

			// ------------------------------------------------
			// Update along the columns
			// ------------------------------------------------
			// Forward Elimination: parallelizable by column; each column of the grid represents an independent 
			// independent tridiagonal. Updates along a row depend on the previous value in the same column, 
			// so it is not parallelizable along rows, but can be parallel between different columns.
			blockLayout = dim3(BLOCK_SIZE, 1, 1);
			gridLayout 	= dim3(nBlocks, 1, 1);
			kernel_row_forward_elimination<<<gridLayout, blockLayout>>>(n, X, A, B);
			gpuErrchk(cudaPeekAtLastError());  
			gpuErrchk(cudaDeviceSynchronize());
			// Normalization: parallelizable by column; each column is independent.
			kernel_row_norm<<<gridLayout, blockLayout>>>(n, X, B);
			gpuErrchk(cudaPeekAtLastError());  
			gpuErrchk(cudaDeviceSynchronize());
			// Back Substitution: parallelizable by column; similar to forward elimination, 
			// each column represents an independent tridiagonal system.
			kernel_row_back_sostitution<<<gridLayout, blockLayout>>>(n, X, A, B);
			gpuErrchk(cudaPeekAtLastError());  
			gpuErrchk(cudaDeviceSynchronize());
		}    
		
		clock_gettime(CLOCK_REALTIME, rt + 1);

		double wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
		printf("ADI (GPU): %9.3f sec\n", wt);
	}

	int compare = compare_matrices(X, X_copy, n);
	if (compare == EXIT_SUCCESS)
		printf("La matrice delle soluzione X è CORRETTA!\n");
	else
		printf("La matrice delle soluzione X è SBAGLIATA!\n");

	// Free memory
	free(X_copy);
	free(B_copy);
	gpuErrchk(cudaFree(X));
	gpuErrchk(cudaFree(B));
	gpuErrchk(cudaFree(A));
	return 0;
}
