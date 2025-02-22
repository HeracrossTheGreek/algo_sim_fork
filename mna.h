#ifndef MNA_H
#define MNA_H

#include "structs.h"

extern double* A_array; // The square matrix A we want to fill
extern double* b_array; // The array b we want to fill
extern double* C_array; // Square matrix C, used in transient analysis
extern int m2_count; // Variable to count how many elements V and L we have inserted
extern int A_dim; // The dimension of the A matrix and b vector

double* alloc_A_array();
double* alloc_b_array();
double* alloc_C_array();
void fill_arrays();
void print_arrays();
void create_equations();
void free_A_array();
void free_b_array();
void free_C_array();

void fill_with_i(component* current);
void fill_with_r(component* current);
void fill_with_v(component* current);
void fill_with_c(component* current);
void fill_with_l(component* current);

#endif
