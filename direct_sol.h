#ifndef DIRECT_SOL_H
#define DIRECT_SOL_H

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>

#include "csparse.h"

extern gsl_matrix* gsl_LU;      // LU array
extern gsl_matrix* gsl_chol;    // Chol array
extern gsl_permutation *gsl_p;  // Permutation

void form_LU();
void form_chol();
void solve_sparse_lu(cs* sparse_cc_A, gsl_vector* cur_gsl_b, gsl_vector *cur_gsl_x);
void solve_sparse_chol(cs* sparse_cc_A, gsl_vector* cur_gsl_b, gsl_vector *cur_gsl_x);

void solve_LU_with_args(gsl_matrix *gsl_A, gsl_vector *gsl_x, gsl_vector *gsl_b);
void solve_chol_with_args(gsl_matrix *gsl_A, gsl_vector *gsl_x, gsl_vector *gsl_b);

#endif