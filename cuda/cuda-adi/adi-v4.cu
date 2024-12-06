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
  int return_value = 1;
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      if (fabs(X[i * n + j] - X_copy[i * n + j]) > 1e-6)
      {
        printf("Mismatch at (%d, %d): Host = %f, Device = %f\n", i, j, X[i * n + j], X_copy[i * n + j]);
        return_value = 0;
      }
    }
  }
  return return_value;
}

void host_adi_col_forward_elimination(int n, DATA_TYPE *X, const DATA_TYPE *A, DATA_TYPE *B)
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
void host_adi_col_norm(int n, DATA_TYPE *X, const DATA_TYPE *B)
{
	for (int col = 0; col < n; col++)
	{
		int idx = col * n + (n - 1);
		if(B[idx] != 0.f)
			X[idx] /= B[idx];
	}
}
void host_adi_col_back_sostitution(int n, DATA_TYPE *X, const DATA_TYPE *A, const DATA_TYPE *B)
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
void host_adi_row_forward_elimination(int n, DATA_TYPE *X, const DATA_TYPE *A, DATA_TYPE *B)
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
void host_adi_row_norm(int n, DATA_TYPE *X, const DATA_TYPE *B)
{
	for (int col = 0; col < n; col++)
	{
		int idx = (n - 1) * n + col;
		if(B[idx] != 0.f)
			X[idx] /= B[idx];
	}
}
void host_adi_row_back_sostitution(int n, DATA_TYPE *X, const DATA_TYPE *A, DATA_TYPE *B)
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
void host_adi(int tsteps, int n, DATA_TYPE *X, const DATA_TYPE *A, DATA_TYPE *B)
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
	 * 1. Ogni blocco è monodimensionale lungo l'asse y. 
	 * 		Quindi, ci sono BLOCK_SIZE thread organizzati verticalmente per blocco.
	 * 2. La riga su cui opera ogni thread è calcolata come: row = blockIdx.y * blockDim.y + threadIdx.y;
	 * 
	 * Gli elementi di X, A, e B sono utilizzati più volte all'interno dello stesso blocco di thread, 
	 * possono essere caricati nella shared memory. 
	 * L'utilizzo della shared memory riduce il numero di accessi alla memoria globale, 
	 * minimizzando il tempo di latenza.
	 */
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < n)
	{
		// Per ogni riga, il thread itera su tutte le colonne da sinistra a destra
		for (int col = 1; col < n; col++)
		{
			// Gli aggiornamenti per X e B sono effettuati in base ai valori precedenti nella stessa riga. 
			// Questo dipende dalla struttura del problema, che ha una dipendenza direzionale lungo le colonne.
			int idx = row * n + col;
			int prev_idx = row * n + (col - 1);
			if(B[prev_idx] != 0.f)
			{
				X[idx] -= X[prev_idx] * A[idx] / B[prev_idx];
				B[idx] -= A[idx] * A[idx] / B[prev_idx];
			}
		}
	}
	// TODO: implementazione con shared memory
	// ...
}
__global__ void kernel_column_norm(int n, DATA_TYPE *X, const DATA_TYPE *B)
{
	/**
	 * Il kernel kernel_column_norm ha lo scopo di normalizzare l'ultima colonna 
	 * della matrice X rispetto a B per ogni riga.
	 * 
	 * !!IMPORTANTE!!
	 * Questo kernel agisce su una singola colonna (n-1). 
	 * L'uso della shared memory può essere evitato, dato che non si eseguono calcoli iterativi complessi sulla stessa riga.
	 * Aggiunge latenza con l'uso di __syncthreads() e potrebbe peggiorare le prestazioni
	 */

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < n) 
	{
		// Data la riga viene calcolato l'indice dell'ultima colonna
		// L'elemento X[last_col_idx] viene normalizzato dividendo per B[last_col_idx]
		// Ogni thread normalizza un elemento nell'ultima colonna di una riga.
		int last_col_idx = row * n + (n - 1);
		if (B[last_col_idx] != 0.f) // impossibile dividere per 0
			X[last_col_idx] /= B[last_col_idx];
	}
}
__global__ void kernel_column_back_sostitution(int n, DATA_TYPE *X, const DATA_TYPE *A, const DATA_TYPE *B)
{
	/**
	 * Il kernel kernel_column_back_sostitution implementa la sostituzione all'indietro 
	 * lungo le colonne per ogni riga della matrice X. 
	 * Ogni thread elabora una riga indipendente dalle altre.
	 * 
	 * È possibile utilizzare la shared memory per evitare accessi ripetuti alla memoria globale. 
	 * Ad esempio, si potrebbe caricare una riga di X, A e B in shared memory
	 */

	// Assegno una riga a ogni thread in base a blockIdx.y e threadIdx.y
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < n) 
	{
		// Parte dalla penultima colonna (n - 2) e procede verso sinistra, 
		// poiché la sostituzione all'indietro si basa sui valori calcolati nelle colonne successive
		for (int col = n - 2; col >= 0; col--) 
		{
			int idx = row * n + col;
			int next_idx = row * n + (col + 1);
			if (B[idx] != 0.f)
				X[idx] = (X[idx] - X[next_idx] * A[next_idx]) / B[idx];
		}
	}

	// TODO: implementazione con shared memory
	// ...
}
__global__ void kernel_row_forward_elimination(int n, DATA_TYPE *X, const DATA_TYPE *A, DATA_TYPE *B)
{
	/**
	 * Il kernel implementa l'eliminazione in avanti lungo le righe, operando su ogni colonna della matrice X. 
	 * L'approccio adottato assegna a ciascun thread una colonna e parallelizza l'elaborazione lungo l'asse delle colonne.
	 * 
	 * Ogni thread lavora su una colonna indipendentemente dalle altre, 
	 * rendendo il kernel parallelizzabile lungo l'asse x. 
	 * Inoltre L'uso della memoria globale per X, A, e B per ogni accesso può risultare inefficiente.
	 * L'uso della shared memory potrebbe migliorare l'efficienza del kernel, riducendo gli accessi alla memoria globale.
	 */

	// Associo ogni thread a una colonna, utilizzando blockIdx.x e threadIdx.x
	// Calcolo dell'Indice della colonna come: col = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < n)
	{
		// L'eliminazione in avanti procede dalla seconda riga (row = 1). 
		// La riga corrente dipende dai valori calcolati nella riga precedente (row - 1).
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
	 * Il kernel si occupa della normalizzazione dell'ultima riga della matrice X,
	 * dividendo ciascun elemento della riga per il corrispondente elemento della matrice B, sempre sull'ultima riga.
	 * 
	 * In questo caso, l'uso della shared memory non è vantaggioso. 
	 * Poiché il kernel opera solo su una singola riga della matrice e ciascun thread elabora un elemento indipendente.
	 */

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < n)
	{
		// Questo calcolo determina l'indice globale per l'elemento col della ultima riga di X e B.
		int last_row_idx = (n - 1) * n + col;
		if(B[last_row_idx] != 0.f)
			X[last_row_idx] /= B[last_row_idx];
	}
}
__global__ void kernel_row_back_sostitution(int n, DATA_TYPE *X, const DATA_TYPE *A, const DATA_TYPE *B)
{
	/**
	 * Il kernel kernel_row_back_sostitution si occupa della sostituzione all'indietro (back substitution) 
	 * lungo le righe di una matrice. 
	 * Ogni thread elabora una colonna specifica della matrice X, aggiornando iterativamente i valori di X 
	 * risalendo dalla penultima riga (n - 2) fino alla prima (row = 0).
	 * 
	 * Ogni thread elabora una colonna indipendente della matrice. 
	 * Questo garantisce una parallelizzazione lungo l'asse x.
	 * 
	 * L'uso della shared memory in questo kernel può migliorare le prestazioni, 
	 * poiché consente di ridurre l'accesso ripetuto alla memoria globale.
	 */

	// Ogni thread è responsabile di una colonna della matrice, determinata da blockIdx.x e threadIdx.x
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < n)
	{
		// La sostituzione all'indietro parte dalla penultima riga (n - 2) e risale fino alla prima riga (row = 0).
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
	const int tsteps = TSTEPS;
	const int bytes = sizeof(DATA_TYPE) * n * n;
	struct timespec rt[2];

	// The following data are needed on the GPU side:
	// - X[] read/write
	// - B[] read/write
	// - A[] read only
	// So we can use the unified memory for these 2 variables
	// X=d_X
	// B=d_B
	// While A can be copyied to GPU with cudaMalloc

	DATA_TYPE *X, *B;
	gpuErrchk(cudaMallocManaged(&X, bytes));
	gpuErrchk(cudaMallocManaged(&B, bytes));
	DATA_TYPE *A_dev;
	gpuErrchk(cudaMalloc(&A_dev, bytes));

	DATA_TYPE* A_host = (DATA_TYPE*)malloc(bytes);
	DATA_TYPE* X_copy = (DATA_TYPE*)malloc(bytes);
	DATA_TYPE* B_copy = (DATA_TYPE*)malloc(bytes);

	#pragma omp parallel
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			int idx 			= i * n + j;
			X[idx] 				= ((DATA_TYPE)i * (j + 1) + 1) / n;
			B[idx] 				= ((DATA_TYPE)i * (j + 3) + 3) / n;
			A_host[idx] 	= ((DATA_TYPE)i * (j + 2) + 2) / n;
			X_copy[idx] 	= ((DATA_TYPE)i * (j + 1) + 1) / n;
			B_copy[idx] 	= ((DATA_TYPE)i * (j + 3) + 3) / n;
		}
	}
	gpuErrchk(cudaMemcpy(A_dev, A_host, bytes, cudaMemcpyHostToDevice));

	// call ADI on host
	{
		clock_gettime(CLOCK_REALTIME, rt);
		host_adi(tsteps, n, X_copy, A_host, B_copy);
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
			kernel_column_forward_elimination<<<gridLayout, blockLayout>>>(n, X, A_dev, B);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
			// Normalization: parallelizable per row; each row is independent.
			kernel_column_norm<<<gridLayout, blockLayout>>>(n, X, B);
			gpuErrchk(cudaPeekAtLastError());  
			gpuErrchk(cudaDeviceSynchronize());
			// Back Substitution: parallelizable per row; again, each row represents an independent tridiagonal system.
			// The operation along columns depends on the previous values of the same row.
			kernel_column_back_sostitution<<<gridLayout, blockLayout>>>(n, X, A_dev, B);
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
			kernel_row_forward_elimination<<<gridLayout, blockLayout>>>(n, X, A_dev, B);
			gpuErrchk(cudaPeekAtLastError());  
			gpuErrchk(cudaDeviceSynchronize());
			// Normalization: parallelizable by column; each column is independent.
			kernel_row_norm<<<gridLayout, blockLayout>>>(n, X, B);
			gpuErrchk(cudaPeekAtLastError());  
			gpuErrchk(cudaDeviceSynchronize());
			// Back Substitution: parallelizable by column; similar to forward elimination, 
			// each column represents an independent tridiagonal system.
			kernel_row_back_sostitution<<<gridLayout, blockLayout>>>(n, X, A_dev, B);
			gpuErrchk(cudaPeekAtLastError());  
			gpuErrchk(cudaDeviceSynchronize());
		}    
		
		clock_gettime(CLOCK_REALTIME, rt + 1);

		double wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
		printf("ADI (GPU): %9.3f sec\n", wt);
	}


	if (compare_matrices(X, X_copy, n) && compare_matrices(B, B_copy, n))
	{
		printf("Risultati Host e Device CORRETTI!\n");
	}
	else
	{
		printf("Risultati Host e Device NON corrispondono!\n");
	}

	// Liberazione memoria

	free(X_copy);
	free(B_copy);
	free(A_host);
	gpuErrchk(cudaFree(X));
	gpuErrchk(cudaFree(B));
	gpuErrchk(cudaFree(A_dev));
	return 0;
}
