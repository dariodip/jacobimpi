#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <utils.h>
#include <matrix_utils.h>

#define MAX_ITERATIONS 100
#define DEBUG 0
#define THRESHOLD 1.0e-4
#define MERGE 1


int main(int argc, char **argv) {
  int rank;                           // rank of current process
  int num_proc;                       // processors count
  double time = 0.0;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Status mpi_status; // status

  if (argc != 2 && rank == 0) {
    fprintf(stderr, "usage: mpirun -np num_proc main N \n");
    return -1;
  }
  const int gridsize = atoi(argv[1]); // size of grid
  const int procgridsize = gridsize / num_proc; // size of process grid

  int additional_rows = 2;
  if (rank == 0) {
    additional_rows--;
  }
  if (rank == num_proc - 1) {
    additional_rows--;
  }
  int first_local_row = rank == 0 ? 0 : 1;
  int last_local_row =  rank == 0 ? procgridsize - 1 : procgridsize;

  const int extra_rows = gridsize % num_proc;
  int extra_rows_for_balancing = 0;
  int balancing_index = num_proc - extra_rows;
  if (rank >= balancing_index) {
    last_local_row ++;
    additional_rows ++;
    extra_rows_for_balancing ++;
  }

  const int first_i = rank == 0 ? first_local_row + 1 : first_local_row; // avoid ghostpoints
  const int last_i = rank == num_proc - 1 ? last_local_row - 1: last_local_row;

  float local[procgridsize + additional_rows][gridsize]; // local array. each process gives its grid's partition
  float xnew[procgridsize + additional_rows][gridsize]; // xnew array

  generate_matrix(first_local_row, last_local_row, gridsize, rank, num_proc, local);

#if DEBUG
  printf("Complete array of %d is: \n", rank);
  print_matrix(procgridsize + additional_rows, gridsize, local);
#endif

  MPI_Barrier(MPI_COMM_WORLD);
  time -= MPI_Wtime();

  // Algorithm starts here
  int iteration_count = 0;
  double diffnorm, global_diffnorm = 1.0;

#if DEBUG
  if (rank == 0) {
    printf("---- STARTING ALGORITHM ---- \n");
  }
#endif

  while (iteration_count++ < MAX_ITERATIONS && global_diffnorm > THRESHOLD) {
    // First: exchange rows using a ring hierarchy
    // CASE 1: I'm the last processor so I don't send the last row
    if (rank < num_proc - 1) {
      MPI_Send(local[last_local_row], gridsize, MPI_FLOAT, rank + 1, 0,
               MPI_COMM_WORLD);
    }
    // CASE 2: I'm not the first processor so I have to reveice my first row
    if (rank > 0) {
      MPI_Recv(local[first_local_row - 1], gridsize, MPI_FLOAT, rank - 1, 0,
               MPI_COMM_WORLD, &mpi_status);
    }
    // CASE 3: I'm not the first processor, so I have to send my first (valid)
    // row
    if (rank > 0) {
      MPI_Send(local[first_local_row], gridsize, MPI_FLOAT, rank - 1, 1,
               MPI_COMM_WORLD);
    }
    // CASE 4: I'm not the last processor, so I have to reveice my last row
    if (rank < num_proc - 1) {
      MPI_Recv(local[last_local_row + 1], gridsize, MPI_FLOAT, rank + 1, 1,
               MPI_COMM_WORLD, &mpi_status);
    }
#if DEBUG
    printf("------- ITERATION [%d] --------\n", iteration_count);
    printf("Complete array of %d at iteration %d is: \n", rank, iteration_count);
    print_matrix(procgridsize + additional_rows, gridsize, local);
#endif

    // Do dirty job
    diffnorm = 0.0;

#if DEBUG
    printf("[%d] iteration: %d; first_i: %d; last_i: %d \n", rank, iteration_count, first_i, last_i);
#endif

    for (int i = first_i; i <= last_i; i++) {
      for (int j = 1; j < gridsize - 1; j++) {
        xnew[i][j] = (local[i][j + 1] + local[i][j - 1] 
            + local[i + 1][j] +local[i - 1][j]) / 4.0;
        diffnorm += (xnew[i][j] - local[i][j]) * (xnew[i][j] - local[i][j]);
      }
    }
    for (int i = first_i; i <= last_i; i++) {
      for (int j = 1; j < gridsize - 1; j++) {
        local[i][j] = xnew[i][j];
      }
    }
    MPI_Allreduce(&diffnorm, &global_diffnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    global_diffnorm = sqrt(global_diffnorm);
  }

// Print whole matrix if required
#if MERGE
// TODO verificare che ogni processore abbia size e displacement corretti
// per fare questo creare un array contenente, per ogni rank i, il numero di elementi di 
// competenza di i
  float *local_as_row = NULL;                              // local to send
  float *global_as_row = NULL;                             // global to print
  int array_size = gridsize * (procgridsize + extra_rows_for_balancing); // size of local matrix

  int *displs = (int *)malloc(num_proc*sizeof(int));       // displacement for each local matrix
  int *rcounts = (int *)malloc(num_proc*sizeof(int));     // send count for each local matrix
  displs[0] = 0;                                          // first will be the first
  rcounts[0] = gridsize * procgridsize;                   // and takes always the same
  for (int i = 1; i < num_proc; i++) {                    // i = processor i
    displs[i] = displs[i-1] + rcounts[i-1];               // displace based on preceding
    int odds = (i >= balancing_index) ? gridsize : 0;     // this should take also another row
    rcounts[i] = array_size + odds;                       // local size
  }

  local_as_row = (float *) calloc(array_size, sizeof(float));     // allocate and fill local as a row
  int k = 0;
  for (int i = first_local_row; i <= last_local_row; i++) {
    for (int j = 0; j < gridsize; j++) {
      local_as_row[k++] = local[i][j];
    }
  }

#if DEBUG
    printf("Local as row for %d: \n", rank);          // just a check..
    print_array(local_as_row, array_size);
    fflush(stdout);
#endif

  if (rank == 0) {                        // allocate global matrix (as a row)
    global_as_row = (float *) calloc(gridsize * gridsize, sizeof(float));
  } 

  // merge global matrix (we used Gatherv because of extra rows)
  MPI_Gatherv(local_as_row, array_size, MPI_FLOAT, global_as_row, rcounts,
   displs,  MPI_FLOAT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("Final global matrix \n");
    int k = 0;
    for (int i = 0; i < gridsize; i++) {
      for (int j = 0; j < gridsize; j++) {
        printf("%05.2f ", global_as_row[k++]);
      }
      printf("\n");
      fflush(stdout);
    }
  }

#endif

  MPI_Barrier(MPI_COMM_WORLD);
  time += MPI_Wtime();
  if (rank == 0) {
    printf("Global Diffnorm: %.4f. \n Total iterations: %d. \n", global_diffnorm, iteration_count - 1);
    printf("Time elapsed: %f. \n", time);
    fflush(stdout);
  }

  free(global_as_row);
  free(local_as_row);
  MPI_Finalize();

  return 0;
}
