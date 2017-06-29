#include <stdio.h>

void generate_matrix(int f_row, int l_row, int cols, int rank, int num_proc,
                     float matrix[][cols]) {
  int k = 0;
  for (int i = f_row; i <= l_row; i++) {
    for (int j = 0; j < cols; j++) {
      matrix[i][j] = (++k % (num_proc + 1)) + rank;
    }
  }
}

void print_matrix(int rows_count, int cols_count,float matrix[rows_count][cols_count]) {
  for (int i = 0; i < rows_count; i++) {
    printf("%d |", i);
    for (int j = 0; j < cols_count; j++) {
      printf("%.2f ", matrix[i][j]);
    }
    printf("\n");
  }
  printf("----------- \n");
}
