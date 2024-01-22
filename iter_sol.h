#ifndef ITER_SOL_H
#define ITER_SOL_H

#define EPS 1e-14

#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>

#include "csparse.h"

void solve_cg(gsl_matrix *gsl_A, gsl_vector* cur_gsl_b, gsl_vector *cur_gsl_x);
void solve_bicg(gsl_matrix *gsl_A, gsl_vector* cur_gsl_b, gsl_vector* cur_gsl_x);
gsl_vector* preconditioner_solve(gsl_matrix *gsl_A, gsl_vector *gsl_r, gsl_vector *gsl_z);
void solve_sparse_cg(cs *sparse_cc_A, gsl_vector* cur_gsl_b, gsl_vector* cur_gsl_x); 
void solve_sparse_bicg(cs* sparse_cc_A, gsl_vector* cur_gsl_b, gsl_vector* cur_gsl_x);
int cs_gaxpy_with_gsl_x (const cs *A, gsl_vector* cur_gsl_x, gsl_vector *gsl_y);
int cs_gaxpy_with_gsl_x_trans (const cs *A, gsl_vector* cur_gsl_x, gsl_vector *gsl_y);
gsl_vector* preconditioner_solve_sparse(gsl_vector *gsl_r, gsl_vector *gsl_z, const double *diag_a);
double* find_diag_a(const cs *sparse_C);



#endif
