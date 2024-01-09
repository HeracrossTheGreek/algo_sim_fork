#ifndef SPARSE_SOL_H
#define SPARSE_SOL_H

#include "structs.h"
#include "csparse.h"

extern cs *sparse_A;
extern cs *sparse_C;
extern int sparse_m2_count;
extern int N; //dimension
extern double *b_array_sparse;
extern double *x_array_sparse;

void form_sparse();
void sparse_fill_with_r(component *current, int *nonzero_counter);
void sparse_fill_with_v(component *current, int *nonzero_counter);
void sparse_fill_with_l(component *current, int *nonzero_counter);
void sparse_fill_with_i(component *current);
void create_b_array_for_sparse();
void add_to_sparse_A(int k, int i, int j, double x);
void print_sparse_arrays();
void print_sparse_x();

#endif
