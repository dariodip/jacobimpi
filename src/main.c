#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <utils.h>
#include <matrix_utils.h>

#define MAX_ITERATIONS 100
#define DEBUG 0
#define THRESHOLD 1.0e-4


int main(int argc, char **argv) {

  if (argc != 2) {
    fprintf(stderr, "usage: mpirun -np num_proc main N \n");
    return -1;
  }

  const int gridsize = atoi(argv[1]); // size of grid
  int rank;                           // rank of current process
  int num_proc;                       // processors count
  double time = 0.0;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Status mpi_status; // status

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

  const int extra_rows = rank == num_proc -1 ? gridsize % num_proc : 0;
  if (rank == num_proc - 1 && extra_rows != 0) {
    additional_rows += extra_rows;
    last_local_row += extra_rows;
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


  MPI_Barrier(MPI_COMM_WORLD);
  time += MPI_Wtime();
  if (rank == 0) {
    printf("Global Diffnorm: %.4f. \n Total iterations: %d. \n", global_diffnorm, iteration_count);
    printf("Time elapsed: %f. \n", time);
  }

  MPI_Finalize();

  return 0;
}
