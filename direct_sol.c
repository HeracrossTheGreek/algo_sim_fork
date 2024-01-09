#include "direct_sol.h"
#include "gsl.h"
#include <stdio.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>
#include "mna.h"
#include "structs.h"
#include "parse.h"
#include "sparse_sol.h"

#include <time.h>


gsl_matrix* gsl_LU = NULL;
gsl_matrix* gsl_chol = NULL;
gsl_permutation *gsl_p = NULL;
css *css_S = NULL;
csn *csn_N = NULL;


// This function forms the LU decomposition. 
// The array A is copied to a new array LU, which will have its elements changed to match the LU decomp.
// A gsl permutation is used

void form_LU() {

    gsl_p = gsl_permutation_alloc(A_dim);
    int signnum;

    gsl_LU = gsl_matrix_alloc(A_dim, A_dim);
    gsl_matrix_memcpy(gsl_LU, gsl_A);

    gsl_linalg_LU_decomp(gsl_LU, gsl_p, &signnum);

}

// This function forms the cholesky decomposition
// IMPORTANT: The array that is produced, contains A's elements in the upper triangular section.
// So, only the lower triangular part (including the diagonal) contains elements of the new decomp.

void form_chol() {

    gsl_set_error_handler(&gslErrorHandler);    // Handle error for non-spd matrices

    int chol_success;

    gsl_chol = gsl_matrix_alloc(A_dim, A_dim);
    gsl_matrix_memcpy(gsl_chol, gsl_A);

    chol_success = gsl_linalg_cholesky_decomp1(gsl_chol);

    if (chol_success != GSL_EDOM) {
        spd = TRUE;
    }

}

void solve_sparse_lu(double *cur_b_array_sparse, double *cur_x_array_sparse){


    css_S = cs_sqr(sparse_C, 2, 0);

    csn_N = cs_lu(sparse_C, css_S, 1);


    double *x;
    x = (double *) malloc(sizeof(double)* N);
    if (x == NULL) {
        printf("Error. Memory allocation problems. Exiting..\n");
        exit(EXIT_FAILURE);
    }

    cs_ipvec(N, csn_N->Pinv, cur_b_array_sparse, x);

    cs_lsolve(csn_N->L, x);

    cs_usolve(csn_N->U, x);

    cs_ipvec(N, css_S->Q, x, cur_x_array_sparse);

    //free(x);
}


void solve_sparse_chol(double *cur_b_array_sparse, double *cur_x_array_sparse){

    double *x;

    css_S = cs_schol(sparse_C, 1);
    csn_N = cs_chol(sparse_C, css_S);

    x = (double *) malloc(sizeof(double)*N);
    if (x == NULL) {
        printf("Error. Memory allocation problems. Exiting..\n");
        exit(EXIT_FAILURE);
    }

    cs_ipvec(N, css_S->Pinv, cur_b_array_sparse, x);
    cs_lsolve(csn_N->L, x);
    cs_ltsolve(csn_N->L, x);
    cs_pvec(N, css_S->Pinv, x, cur_x_array_sparse);

    free(x);
}