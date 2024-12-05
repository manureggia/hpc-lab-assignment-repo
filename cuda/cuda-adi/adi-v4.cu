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
int compare_matrices(DATA_TYPE *X_host, DATA_TYPE *X_copyice, int n)
{
  int return_value = 1;
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      if (fabs(X_host[i * n + j] - X_copyice[i * n + j]) > 1e-6)
      {
        printf("Mismatch at (%d, %d): Host = %f, Device = %f\n", i, j, X_host[i * n + j], X_copyice[i * n + j]);
        return_value = 0;
      }
    }
  }
  return return_value;
}

// Kernel host
void kernel_adi_host(int tsteps, int n, DATA_TYPE *X, const DATA_TYPE *A, DATA_TYPE *B)
{
	/**
	 * Questo codice implementa una risoluzione dell'algoritmo Alternating-Direction Implicit (ADI) 
	 * su una griglia bidimensionale. È strutturato per aggiornare le soluzioni delle equazioni 
	 * differenziali in due passaggi: uno lungo le colonne e uno lungo le righe della griglia
	 */

	/** Ad ogni iterazione, il sistema viene aggiornato lungo le colonne e le righe */
	for (int t = 0; t < tsteps; t++)
	{
		/**
		 * Aggiornamento lungo le colonne:
		 * Per ogni riga:
		 * 1. si applica l'eliminazione in avanti per ridurre il sistema tridiagonale lungo le colonne.
		 * 2. X rappresenta il vettore delle soluzioni.
		 * 3. A e B rappresentano i coefficienti delle equazioni differenziali.
		 */
		for (int row = 0; row < n; row++)
		{
			for (int col = 1; col < n; col++)
			{
				X[row * n + col] -= X[row * n + col - 1] * A[row * n + col] / B[row * n + col - 1];
				B[row * n + col] -= A[row * n + col] * A[row * n + col] / B[row * n + col - 1];
			}
		}

		/**
		 * Normalizzazione:
		 * Il valore della soluzione nella parte inferiore della colonna è normalizzato dividendo 
		 * per il coefficiente B
		 */
		for (int col = 0; col < n; col++)
			X[col * n + (n - 1)] /= B[col * n + (n - 1)];

		
		/**
		 * Sostituzione all'indietro (Back Substitution):
		 * Dopo l'eliminazione in avanti, si risolvono i valori risalendo lungo la colonna
		 */
		for (int row = 0; row < n; row++)
			for (int col = 0; col < n - 2; col++)
				X[row * n + (n - col - 2)] = (X[row * n + (n - col - 2)] - X[row * n + (n - col - 3)] * A[row * n + (n - col - 3)]) / B[row * n + (n - col - 3)];
		

		/**
		 * Aggiornamento lungo le righe:
		 * Qui il processo si applica lungo le righe.
		 * Stesso approccio dell'eliminazione lungo le colonne, ma con iterazione spaziale lungo i1.
		 */
		for (int row = 1; row < n; row++)
		{
			for (int col = 0; col < n; col++)
			{
				X[row * n + col] -= X[(row - 1) * n + col] * A[row * n + col] / B[(row - 1) * n + col];
				B[row * n + col] -= A[row * n + col] * A[row * n + col] / B[(row - 1) * n + col];
			}
		}

		/**
		 * Normalizzazione:
		 * Si normalizza l'ultima riga dividendo per B.
		 */
		for (int col = 0; col < n; col++)
			X[(n - 1) * n + col] /= B[(n - 1) * n + col];

		/**
		 * Back-substitution:
		 * Anche in questo caso, si risolve il sistema risalendo lungo le righe
		 */
		for (int row = 0; row < n - 2; row++)
			for (int col = 0; col < n; col++)
				X[(n - 2 - row) * n + col] = (X[(n - 2 - row) * n + col] - X[(n - row - 3) * n + col] * A[(n - 3 - row) * n + col]) / B[(n - 2 - row) * n + col];
	}
}

__global__ void kernel_column_forward_elimination(int n, DATA_TYPE *X, const DATA_TYPE *A, DATA_TYPE *B)
{
	// SENZA SHARED MEMORY
	// -----------------------------------------------
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < n)
	{
		for (int col = 1; col < n; col++) 
		{
			int idx = row * n + col;
			int prev_idx = row * n + (col - 1);
			X[idx] -= X[prev_idx] * A[idx] / B[prev_idx];
			B[idx] -= A[idx] * A[idx] / B[prev_idx];
		}
	}

	/**
	 * Gli elementi di X, A, e B sono utilizzati più volte all'interno dello stesso blocco di thread, 
	 * possono essere caricati nella shared memory. La shared memory riduce il numero di accessi alla memoria globale, 
	 * minimizzando il tempo di latenza.
	 */

	// todo ...
}
__global__ void kernel_column_norm(int n, DATA_TYPE *X, const DATA_TYPE *B)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < n) 
	{
		int last_col_idx = row * n + (n - 1);
		X[last_col_idx] /= B[last_col_idx];
	}
}
__global__ void kernel_column_back_sostitution(int n, DATA_TYPE *X, const DATA_TYPE *A, const DATA_TYPE *B)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < n) 
	{
		for (int col = n - 2; col >= 0; col--) 
		{
			int idx = row * n + col;
			int next_idx = row * n + (col + 1);
			X[idx] = (X[idx] - X[next_idx] * A[next_idx]) / B[idx];
		}
	}
}
__global__ void kernel_row_forward_elimination(int n, DATA_TYPE *X, const DATA_TYPE *A, DATA_TYPE *B)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < n)
	{
		for (int row = 1; row < n; row++) 
		{
			int idx = row * n + col;
			int prev_idx = (row - 1) * n + col;
			X[idx] -= X[prev_idx] * A[idx] / B[prev_idx];
			B[idx] -= A[idx] * A[idx] / B[prev_idx];
		}
	}
}
__global__ void kernel_row_norm(int n, DATA_TYPE *X, const DATA_TYPE *B)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < n)
	{
		int last_row_idx = (n - 1) * n + col;
		X[last_row_idx] /= B[last_row_idx];
	}
}
__global__ void kernel_row_back_sostitution(int n, DATA_TYPE *X, const DATA_TYPE *A, const DATA_TYPE *B)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < n)
	{
		for (int row = n - 2; row >= 0; row--) 
		{
			int idx = row * n + col;
			int next_idx = (row + 1) * n + col;
			X[idx] = (X[idx] - X[next_idx] * A[next_idx]) / B[idx];
		}
	}
}

int main()
{
	const int n = N;
	const int tsteps = TSTEPS;
	const int bytes = sizeof(DATA_TYPE) * n * n;
	struct timespec rt[2];

	// Lato GPU sono necessari i seguenti dati:
	// - X[] lettura/scrittura
	// - B[] lettura/scrittura
	// - A[] solo lettura
	// Quindi possiamo usare la memoria unificata per queste 3 variabili
	// X=d_X
	// B=d_B
	// A=d_A

	DATA_TYPE *X, *A, *B;
	gpuErrchk(cudaMallocManaged(&A, bytes));
	gpuErrchk(cudaMallocManaged(&X, bytes));
	gpuErrchk(cudaMallocManaged(&B, bytes));

	DATA_TYPE* X_copy = (DATA_TYPE*)malloc(bytes);
	DATA_TYPE* B_copy = (DATA_TYPE*)malloc(bytes);

	#pragma omp parallel
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			int idx 			= i * n + j;
			X[idx] 				= ((DATA_TYPE)i * (j + 1) + 1) / n;
			A[idx] 				= ((DATA_TYPE)i * (j + 2) + 2) / n;
			B[idx] 				= ((DATA_TYPE)i * (j + 3) + 3) / n;
			X_copy[idx] 	= ((DATA_TYPE)i * (j + 1) + 1) / n;
			B_copy[idx] 	= ((DATA_TYPE)i * (j + 3) + 3) / n;
		}
	}

	// call ADI on host
	{
		clock_gettime(CLOCK_REALTIME, rt);
		kernel_adi_host(tsteps, n, X_copy, A, B_copy);
		clock_gettime(CLOCK_REALTIME, rt + 1);

		double wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
		printf("ADI (Host): %9.3f sec\n", wt);
	}

	// call ADI on GPU
	{
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

		clock_gettime(CLOCK_REALTIME, rt);
		/**
		 * Come abbiamo visto nell'implementazione lato host, l'algoritmo adi si compone di diverse
		 * passi fondamentali:
		 * [1] Aggiornamento lungo le colonne: 
		 * 		[1.1] eliminazione in avanti (Forward Elimination)
		 * 		[1.2] normalizzazione
		 * 		[1.3] sostituzione all'indietro (Back Substitution)
		 * [2] Aggiornamento lungo le righe:
		 * 		[2.1] eliminazione in avanti
		 * 		[2.2] normalizzazione
		 * 		[2.3] sostituzione all'indietro
		 * 
		 * Nell'algoritmo ADI, alcune operazioni possono essere eseguite in parallelo perché non dipendono 
		 * direttamente dai risultati degli altri calcoli per ogni iterazione spaziale.
		 * Le operazioni possono essere parallelizzate per righe durante l'aggiornamento lungo le colonne e 
		 * per colonne durante l'aggiornamento lungo le righe.
		 */
		for (int t = 0; t < tsteps; t++)
		{
			// ------------------------------------------------
			// [1] Aggiornamento lungo le colonne
			// ------------------------------------------------
			// [1.1] eliminazione in avanti (Forward Elimination): 
			// gli aggiornamenti lungo una colonna di una stessa riga dipendono dal valore precedente
			// nella stessa riga, quindi non è parallelizzabile lungo le colonne,
			// ma l'operazione per righe differenti è indipendente.
			kernel_column_forward_elimination<<<grid, block>>>(n, X, A, B);
			gpuErrchk(cudaPeekAtLastError());  
			gpuErrchk(cudaDeviceSynchronize());
			// [1.2] normalizzazione: 
			// parallelizzabile per riga; ogni riga è indipendente.
			kernel_column_norm<<<grid, block>>>(n, X, B);
			gpuErrchk(cudaPeekAtLastError());  
			gpuErrchk(cudaDeviceSynchronize());
			// [1.3] sostituzione all'indietro (Back Substitution):
			// parallelizzabile per riga; anche qui, ogni riga rappresenta un sistema tridiagonale indipendente.
			// L'operazione lungo colonne dipende dai valori precedenti della stessa riga.
			kernel_column_back_sostitution<<<grid, block>>>(n, X, A, B);
			gpuErrchk(cudaPeekAtLastError());  
			gpuErrchk(cudaDeviceSynchronize());

			// ------------------------------------------------
			// [2] Aggiornamento lungo le righe
			// ------------------------------------------------
			// [2.1] eliminazione in avanti:
			// parallelizzabile per colonna; ogni colonna della griglia rappresenta un sistema 
			// tridiagonale indipendente.
			// Gli aggiornamenti lungo una riga dipendono dal valore precedente nella stessa colonna, 
			// quindi non è parallelizzabile lungo le righe, ma può essere parallelo tra colonne diverse.
			kernel_row_forward_elimination<<<grid, block>>>(n, X, A, B);
			gpuErrchk(cudaPeekAtLastError());  
			gpuErrchk(cudaDeviceSynchronize());
			// [2.2] normalizzazione: 
			// parallelizzabile per colonna; ogni colonna è indipendente.
			kernel_row_norm<<<grid, block>>>(n, X, B);
			gpuErrchk(cudaPeekAtLastError());  
			gpuErrchk(cudaDeviceSynchronize());
			// [2.3] sostituzione all'indietro:
			// parallelizzabile per colonna; simile all'eliminazione in avanti, 
			// ogni colonna rappresenta un sistema tridiagonale indipendente.
			kernel_row_back_sostitution<<<grid, block>>>(n, X, A, B);
			gpuErrchk(cudaPeekAtLastError());  
			gpuErrchk(cudaDeviceSynchronize());
		}    
		
		clock_gettime(CLOCK_REALTIME, rt + 1);

		double wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
		printf("ADI (GPU): %9.3f sec\n", wt);
	}


	if (compare_matrices(X, X_copy, n))
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
	gpuErrchk(cudaFree(X));
	gpuErrchk(cudaFree(A));
	gpuErrchk(cudaFree(B));
	return 0;
}
