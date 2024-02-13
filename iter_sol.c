#include "direct_sol.h"
#include <stdio.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_errno.h>
#include "mna.h"
#include "structs.h"
#include "parse.h"
#include "iter_sol.h"
#include "sparse_sol.h"
#include "csparse.h"
#include "math.h"
#include "string.h"
#include "gsl.h"
// Algorithm for the CG algorithm, which is used to solve SPD systems iteratively

void solve_cg(gsl_matrix *gsl_A, gsl_vector* cur_gsl_b, gsl_vector* cur_gsl_x) {

    int max_iter = A_dim;   // A_dim = n

    gsl_vector *gsl_r = gsl_vector_alloc(A_dim);    // residual vector
    gsl_vector *gsl_Ax = gsl_vector_alloc(A_dim);   // vector Ax
    gsl_vector *gsl_z = gsl_vector_alloc(A_dim);    // vector z
    gsl_vector *gsl_p = gsl_vector_alloc(A_dim);    // vector p
    gsl_vector *gsl_q = gsl_vector_alloc(A_dim);    // vector q

    gsl_blas_dgemv(CblasNoTrans, 1.0, gsl_A, cur_gsl_x, 0.0, gsl_Ax);   // Ax = A*x

    gsl_vector_memcpy(gsl_r, cur_gsl_b);
    gsl_vector_sub(gsl_r, gsl_Ax);      // r = b - Ax

    double b_norm = gsl_blas_dnrm2(cur_gsl_b);  // norm of vector b
    double r_norm = gsl_blas_dnrm2(gsl_r);     // norm of vector r
    double rho;
    double rho1 = 1;
    double beta;
    double pq_dot;
    double alpha;
    int i = 0;

    if (b_norm == 0) b_norm = 1;            // If the norm of b is 0, make it 1

    while(r_norm/b_norm > itol && i<2*max_iter) {

        i++;

        gsl_z = preconditioner_solve(gsl_A, gsl_r, gsl_z);    // Solve Mz = r

        gsl_blas_ddot(gsl_r, gsl_z, &rho);   // dot(r,z)

        if (i == 1) {
            gsl_vector_memcpy(gsl_p, gsl_z);    // p = z
        }
        else {
            beta = rho/rho1;
            gsl_vector_axpby(1, gsl_z, beta, gsl_p);    // p = z + beta*p

        }

        rho1 = rho;
        gsl_blas_dgemv(CblasNoTrans, 1.0, gsl_A, gsl_p, 0.0, gsl_q);   // q = A*p

        gsl_blas_ddot(gsl_p, gsl_q, &pq_dot);   // dot(p,q)

        alpha = rho/pq_dot; // alpha = rho/(dot(p,q))

        gsl_blas_daxpy(alpha, gsl_p, cur_gsl_x);        // x = x + alpha*p
        gsl_blas_daxpy(-alpha, gsl_q, gsl_r);           // r = r - alpha*q

        r_norm = gsl_blas_dnrm2(gsl_r); // norm of vector r

    }

    gsl_vector_free(gsl_r);
    gsl_vector_free(gsl_Ax);
    gsl_vector_free(gsl_z);
    gsl_vector_free(gsl_p);
    gsl_vector_free(gsl_q);
    

}

void solve_bicg(gsl_matrix *gsl_A, gsl_vector* cur_gsl_b, gsl_vector* cur_gsl_x) {


    int max_iter = A_dim;   // A_dim = n

    gsl_vector *gsl_r = gsl_vector_alloc(A_dim);        // residual vector
    gsl_vector *gsl_r_bicg = gsl_vector_alloc(A_dim);   // bicg residual vector

    gsl_vector *gsl_Ax = gsl_vector_alloc(A_dim);       // vector Ax
    gsl_vector *gsl_z = gsl_vector_alloc(A_dim);        // vector z
    gsl_vector *gsl_z_bicg = gsl_vector_alloc(A_dim);   // vector z'
    gsl_vector *gsl_p = gsl_vector_alloc(A_dim);        // vector p
    gsl_vector *gsl_p_bicg = gsl_vector_alloc(A_dim);   // vector p'
    gsl_vector *gsl_q = gsl_vector_alloc(A_dim);        // vector q
    gsl_vector *gsl_q_bicg = gsl_vector_alloc(A_dim);   // vector q'

    gsl_blas_dgemv(CblasNoTrans, 1.0, gsl_A, cur_gsl_x, 0.0, gsl_Ax);   // Ax = A*x

    gsl_vector_memcpy(gsl_r, cur_gsl_b);
    gsl_vector_sub(gsl_r, gsl_Ax);              // r = b - Ax

    gsl_vector_memcpy(gsl_r_bicg, cur_gsl_b);
    gsl_vector_sub(gsl_r_bicg, gsl_Ax);         // r' = b - Ax

    double b_norm = gsl_blas_dnrm2(cur_gsl_b);  // norm of vector b
    double r_norm = gsl_blas_dnrm2(gsl_r);   // norm of vector r
    double rho;
    double rho1 = 1;
    double beta;
    double omega;
    double alpha;
    int i = 0;

    if (b_norm == 0) b_norm = 1;            // If the norm of b is 0, make it 1

    while(r_norm/b_norm > itol && i<2*max_iter) {

        i++;

        gsl_z = preconditioner_solve(gsl_A, gsl_r, gsl_z);    // Solve Mz = r
        gsl_z_bicg = preconditioner_solve(gsl_A, gsl_r_bicg, gsl_z_bicg);    // Solve Mz' = r'

        gsl_blas_ddot(gsl_r_bicg, gsl_z, &rho);   // dot(r',z)

        if (fabs(rho) < EPS) {

            break;
        }
        // SEE MATLAB!!
        if (i == 1) {
            gsl_vector_memcpy(gsl_p, gsl_z);    // p = z
            gsl_vector_memcpy(gsl_p_bicg, gsl_z_bicg);    // p' = z'
        }
        else {
            beta = rho/rho1;
            gsl_vector_axpby(1, gsl_z, beta, gsl_p);    // p = z + beta*p
            gsl_vector_axpby(1, gsl_z_bicg, beta, gsl_p_bicg);    // p' = z' + beta*p'

        }

        rho1 = rho;
        gsl_blas_dgemv(CblasNoTrans, 1.0, gsl_A, gsl_p, 0.0, gsl_q);   // q = A*p
        gsl_blas_dgemv(CblasTrans, 1.0, gsl_A, gsl_p_bicg, 0.0, gsl_q_bicg);   // q' = A(T)*p'

        gsl_blas_ddot(gsl_p_bicg, gsl_q, &omega);   // omega = dot(p',q)

        if (fabs(omega) < EPS) {
            break;
        }

        alpha = rho/omega;
        
        gsl_blas_daxpy(alpha, gsl_p, cur_gsl_x);        // x = x + alpha*p
        gsl_blas_daxpy(-alpha, gsl_q, gsl_r);           // r = r - alpha*q
        gsl_blas_daxpy(-alpha, gsl_q_bicg, gsl_r_bicg); // r' = r' - alpha*q'

        r_norm = gsl_blas_dnrm2(gsl_r); // norm of vector r



    }

    gsl_vector_free(gsl_r);
    gsl_vector_free(gsl_r_bicg);
    gsl_vector_free(gsl_Ax);
    gsl_vector_free(gsl_z);
    gsl_vector_free(gsl_z_bicg);
    gsl_vector_free(gsl_p);
    gsl_vector_free(gsl_p_bicg);
    gsl_vector_free(gsl_q);
    gsl_vector_free(gsl_q_bicg);

}

// This function executes the preconditioning process, filling vector gsl_z
// Element i of gsl_z[i] = r[i] * 1/A[i][i]

gsl_vector* preconditioner_solve(gsl_matrix *gsl_A, gsl_vector *gsl_r, gsl_vector *gsl_z) {

    //gsl_vector *gsl_z = gsl_vector_alloc(A_dim);
    double cur_A_element;
    double cur_z_element;

    for (int i=0; i<A_dim; i++) {

        cur_A_element = gsl_matrix_get(gsl_A, i, i);

        if (cur_A_element == 0) cur_A_element = 1;

        cur_z_element = gsl_vector_get(gsl_r, i) / cur_A_element;

        gsl_vector_set(gsl_z, i, cur_z_element);
    }

    return gsl_z;
}

//////////////////////////////////////////////////////////////////////
/////////////////////////// Sparse Solvers ///////////////////////////
//////////////////////////////////////////////////////////////////////

// Sparse implementation of the CG algorithm, which is used to solve SPD systems iteratively
void solve_sparse_cg(cs *sparse_cc_A, gsl_vector* cur_gsl_b, gsl_vector* cur_gsl_x) {

    int max_iter = A_dim;

    gsl_vector *gsl_r = gsl_vector_alloc(A_dim);        // residual vector

    gsl_vector *gsl_Ax = gsl_vector_calloc(A_dim);       // vector Ax
    gsl_vector *gsl_z = gsl_vector_alloc(A_dim);        // vector z
    gsl_vector *gsl_p = gsl_vector_alloc(A_dim);        // vector p
    gsl_vector *gsl_q = gsl_vector_alloc(A_dim);        // vector q

    cs_gaxpy_with_gsl_x(sparse_cc_A, cur_gsl_x, gsl_Ax);   // Ax = A*x

    gsl_vector_memcpy(gsl_r, cur_gsl_b);
    gsl_vector_sub(gsl_r, gsl_Ax);              // r = b - Ax

    double b_norm = gsl_blas_dnrm2(cur_gsl_b);  // norm of vector b
    double r_norm= gsl_blas_dnrm2(gsl_r);       // norm of vector r
    double rho;
    double rho1=1;
    double beta;
    double pq_dot;
    double alpha;
    int i=0;

    double *diag_a = find_diag_a(sparse_cc_A); //used for preconditioner!

    if (b_norm == 0) b_norm = 1;            // If the norm of b is 0, make it 1

    while(r_norm/b_norm > itol && i<2*max_iter) {

        i++;

        if (r_norm/b_norm <= itol) break;

        gsl_z = preconditioner_solve_sparse(gsl_r, gsl_z, diag_a);    // Solve Mz = r

        gsl_blas_ddot(gsl_r, gsl_z, &rho);   // dot(r,z)

        if (i == 1) {
            gsl_vector_memcpy(gsl_p, gsl_z);    // p = z
        }
        else {
            beta = rho/rho1;
            gsl_vector_axpby(1, gsl_z, beta, gsl_p);    // p = z + beta*p

        }

        rho1 = rho;

        gsl_vector_set_zero(gsl_q); // initialize gsl_q to zero, necessary for the next axpy function
        cs_gaxpy_with_gsl_x(sparse_cc_A, gsl_p, gsl_q);  // q = A*p 

        gsl_blas_ddot(gsl_p, gsl_q, &pq_dot);   // omega = dot(p',q)


        alpha = rho/pq_dot;
        
        gsl_blas_daxpy(alpha, gsl_p, cur_gsl_x);        // x = x + alpha*p
        gsl_blas_daxpy(-alpha, gsl_q, gsl_r);           // r = r - alpha*q

        r_norm = gsl_blas_dnrm2(gsl_r); // norm of vector r

    }

    gsl_vector_free(gsl_r);
    gsl_vector_free(gsl_Ax);
    gsl_vector_free(gsl_z);
    gsl_vector_free(gsl_p);
    gsl_vector_free(gsl_q);
    free(diag_a);

}
// Sparse implementation of the BiCG algorithm, which is used to solve non-SPD systems iteratively
void solve_sparse_bicg(cs *sparse_cc_A, gsl_vector* cur_gsl_b, gsl_vector* cur_gsl_x) {

    int max_iter = A_dim;

    gsl_vector *gsl_r = gsl_vector_alloc(A_dim);        // residual vector
    gsl_vector *gsl_r_bicg = gsl_vector_alloc(A_dim);   // bicg residual vector

    gsl_vector *gsl_Ax = gsl_vector_calloc(A_dim);       // vector Ax
    gsl_vector *gsl_z = gsl_vector_alloc(A_dim);        // vector z
    gsl_vector *gsl_z_bicg = gsl_vector_alloc(A_dim);   // vector z'
    gsl_vector *gsl_p = gsl_vector_alloc(A_dim);        // vector p
    gsl_vector *gsl_p_bicg = gsl_vector_alloc(A_dim);   // vector p'
    gsl_vector *gsl_q = gsl_vector_alloc(A_dim);        // vector q
    gsl_vector *gsl_q_bicg = gsl_vector_alloc(A_dim);   // vector q'

    cs_gaxpy_with_gsl_x(sparse_cc_A, cur_gsl_x, gsl_Ax);   // Ax = A*x

    gsl_vector_memcpy(gsl_r, cur_gsl_b);
    gsl_vector_sub(gsl_r, gsl_Ax);              // r = b - Ax

    gsl_vector_memcpy(gsl_r_bicg, cur_gsl_b);
    gsl_vector_sub(gsl_r_bicg, gsl_Ax);         // r' = b - Ax

    double b_norm = gsl_blas_dnrm2(cur_gsl_b);  // norm of vector b
    double r_norm = gsl_blas_dnrm2(gsl_r);      // norm of vector r
    double rho;
    double rho1 = 1;
    double beta;
    double omega;
    double alpha;
    int i = 0;

    double *diag_a = find_diag_a(sparse_cc_A); //used for preconditioner!

    if (b_norm == 0) b_norm = 1;            // If the norm of b is 0, make it 1

    while(r_norm/b_norm > itol && i<2*max_iter) {

        i++;

        if (r_norm/b_norm <= itol) break;

        gsl_z = preconditioner_solve_sparse(gsl_r, gsl_z, diag_a);    // Solve Mz = r
        gsl_z_bicg = preconditioner_solve_sparse(gsl_r_bicg, gsl_z_bicg, diag_a);    // Solve Mz' = r'

        gsl_blas_ddot(gsl_r_bicg, gsl_z, &rho);   // dot(r',z)

        if (fabs(rho) < EPS) break;

        if (i == 1) {
            gsl_vector_memcpy(gsl_p, gsl_z);    // p = z
            gsl_vector_memcpy(gsl_p_bicg, gsl_z_bicg);    // p' = z'
        }
        else {
            beta = rho/rho1;
            gsl_vector_axpby(1, gsl_z, beta, gsl_p);    // p = z + beta*p
            gsl_vector_axpby(1, gsl_z_bicg, beta, gsl_p_bicg);    // p' = z' + beta*p'

        }

        rho1 = rho;

        gsl_vector_set_zero(gsl_q); // initialize gsl_q to 0, necessary for the following axpy function
        cs_gaxpy_with_gsl_x(sparse_cc_A, gsl_p, gsl_q);  // q = A*p 

        cs_gaxpy_with_gsl_x_trans(sparse_cc_A, gsl_p_bicg, gsl_q_bicg);  // q' = A(T)*p'

        gsl_blas_ddot(gsl_p_bicg, gsl_q, &omega);   // omega = dot(p',q)

        if (fabs(omega) < EPS) break;

        alpha = rho/omega;
        
        gsl_blas_daxpy(alpha, gsl_p, cur_gsl_x);        // x = x + alpha*p
        gsl_blas_daxpy(-alpha, gsl_q, gsl_r);           // r = r - alpha*q
        gsl_blas_daxpy(-alpha, gsl_q_bicg, gsl_r_bicg); // r' = r' - alpha*q'

        r_norm = gsl_blas_dnrm2(gsl_r);

    }

    gsl_vector_free(gsl_r);
    gsl_vector_free(gsl_r_bicg);
    gsl_vector_free(gsl_Ax);
    gsl_vector_free(gsl_z);
    gsl_vector_free(gsl_z_bicg);
    gsl_vector_free(gsl_p);
    gsl_vector_free(gsl_p_bicg);
    gsl_vector_free(gsl_q);
    gsl_vector_free(gsl_q_bicg);
    free(diag_a);

}

// This function performs the axpy operation y = A*x
int cs_gaxpy_with_gsl_x(const cs *A, gsl_vector* cur_gsl_x, gsl_vector *gsl_y) {
    int p, j, n, *Ap, *Ai;
    double *Ax;

    n = A->n;
    Ap = A->p;
    Ai = A->i;
    Ax = A->x;

    for (j = 0; j < n; j++) {
        int start = Ap[j];
        int end = Ap[j + 1];
        double xj = gsl_vector_get(cur_gsl_x, j);

        for (p = start; p < end; p++) {
            gsl_vector_set(gsl_y, Ai[p], gsl_vector_get(gsl_y, Ai[p]) + Ax[p] * xj);
        }
    }

    return 1;
}

// This function performs the axpy operation y = Atranspose*x
int cs_gaxpy_with_gsl_x_trans(const cs *A, gsl_vector* cur_gsl_x, gsl_vector *gsl_y) {
    int p, j, n, *Ap, *Ai;
    double *Ax;

    n = A->n;
    Ap = A->p;
    Ai = A->i;
    Ax = A->x;

    for (j = 0; j < n; j++) {
        int start = Ap[j];
        int end = Ap[j + 1];
        double yj = 0.0;

        for (p = start; p < end; p++) {
            yj += Ax[p] * gsl_vector_get(cur_gsl_x, Ai[p]);
        }

        gsl_vector_set(gsl_y, j, yj);
    }

    return 1;
}

// This function solves the preconditioner operation Mz = r, where M is the diagonal of A
// The diagonal of A is already computated

gsl_vector* preconditioner_solve_sparse(gsl_vector *gsl_r, gsl_vector *gsl_z, const double *diag_a) {

    //gsl_vector *gsl_z = gsl_vector_alloc(A_dim);
    double cur_A_element;
    double cur_z_element;

    for (int i=0; i<A_dim; i++) {

        cur_A_element = diag_a[i];
        if (cur_A_element == 0) cur_A_element = 1;

        cur_z_element = gsl_vector_get(gsl_r, i) / cur_A_element;

        gsl_vector_set(gsl_z, i, cur_z_element);
    }

    return gsl_z;
}

// This function calculates the diagonal of A
double* find_diag_a(const cs *sparse_C) {

    int N = sparse_C->n;
    double *diag_a = (double *)malloc(N * sizeof(double));

    for (int j = 0; j < N; j++) {
        diag_a[j] = 0.0;  // Default value for missing diagonal elements

        for (int p = sparse_C->p[j]; p < sparse_C->p[j + 1]; p++) {
            if (sparse_C->i[p] == j) {
                diag_a[j] = sparse_C->x[p];
                break;  // Found the diagonal element, no need to continue
            }
        }
    }

    return diag_a;
}

void solve_bicg_complex(gsl_matrix_complex *cur_gsl_A, gsl_vector_complex* gsl_b, gsl_vector_complex* cur_gsl_x) {

    int max_iter = A_dim;   // A_dim = n

    gsl_vector_complex *gsl_r = gsl_vector_complex_alloc(A_dim);        // residual vector
    gsl_vector_complex *gsl_r_bicg = gsl_vector_complex_alloc(A_dim);   // bicg residual vector

    gsl_vector_complex *gsl_Ax = gsl_vector_complex_alloc(A_dim);       // vector Ax
    gsl_vector_complex *gsl_z = gsl_vector_complex_alloc(A_dim);        // vector z
    gsl_vector_complex *gsl_z_bicg = gsl_vector_complex_alloc(A_dim);   // vector z'
    gsl_vector_complex *gsl_p = gsl_vector_complex_alloc(A_dim);        // vector p
    gsl_vector_complex *gsl_p_bicg = gsl_vector_complex_alloc(A_dim);   // vector p'
    gsl_vector_complex *gsl_q = gsl_vector_complex_alloc(A_dim);        // vector q
    gsl_vector_complex *gsl_q_bicg = gsl_vector_complex_alloc(A_dim);   // vector q'

    gsl_complex complex_0 = gsl_complex_rect(0.0, 0.0);
    gsl_complex complex_1 = gsl_complex_rect(1.0, 0.0);
    
    gsl_blas_zgemv(CblasNoTrans, complex_1, cur_gsl_A, cur_gsl_x, complex_0, gsl_Ax);   // Ax = A*x

    gsl_vector_complex_memcpy(gsl_r, gsl_b);
    gsl_vector_complex_sub(gsl_r, gsl_Ax);              // r = b - Ax

    gsl_vector_complex_memcpy(gsl_r_bicg, gsl_b);
    gsl_vector_complex_sub(gsl_r_bicg, gsl_Ax);         // r' = b - Ax

    double b_norm = gsl_blas_dznrm2(gsl_b);  // norm of vector b
    double r_norm = gsl_blas_dznrm2(gsl_r);   // norm of vector r
    gsl_complex rho;
    gsl_complex rho1 = gsl_complex_rect(1.0, 0.0);
    gsl_complex beta;
    gsl_complex beta_conj;
    gsl_complex omega;
    gsl_complex alpha;
    gsl_complex alpha_conj;
    gsl_complex neg_alpha;
    gsl_complex neg_alpha_conj;
    int i = 0;

    // cur_gsl_A contains the conjugate transpose of A, used in preconditioning
    gsl_matrix_complex *cur_gsl_A_conjtrans = gsl_matrix_complex_alloc(A_dim, A_dim);
    gsl_matrix_complex_conjtrans_memcpy(cur_gsl_A_conjtrans, cur_gsl_A);

    if (b_norm == 0) b_norm = 1;            // If the norm of b is 0, make it 1

    while(r_norm/b_norm > itol && i<2*max_iter) {

        i++;

        gsl_z = preconditioner_solve_complex(cur_gsl_A, gsl_r, gsl_z);    // Solve Mz = r
        gsl_z_bicg = preconditioner_solve_complex(cur_gsl_A_conjtrans, gsl_r_bicg, gsl_z_bicg);    // Solve M(H)z' = r'

        gsl_blas_zdotc(gsl_r_bicg, gsl_z, &rho);   // dot(r'(H),z) (zdotc does this specific dot)

        if (gsl_complex_abs(rho) < EPS) {

            break;
        }

        if (i == 1) {
            gsl_vector_complex_memcpy(gsl_p, gsl_z);    // p = z
            gsl_vector_complex_memcpy(gsl_p_bicg, gsl_z_bicg);    // p' = z'
        }
        else {
            beta = gsl_complex_div(rho, rho1);
            beta_conj = gsl_complex_conjugate(beta);
            gsl_vector_complex_axpby(complex_1, gsl_z, beta, gsl_p);    // p = z + beta*p
            gsl_vector_complex_axpby(complex_1, gsl_z_bicg, beta_conj, gsl_p_bicg);    // p' = z' + beta'*p'

        }

        rho1 = rho;
        gsl_blas_zgemv(CblasNoTrans, complex_1, cur_gsl_A, gsl_p, complex_0, gsl_q);   // q = A*p
        gsl_blas_zgemv(CblasTrans, complex_1, cur_gsl_A_conjtrans, gsl_p_bicg, complex_0, gsl_q_bicg);   // q' = A(H)*p'

        gsl_blas_zdotc(gsl_p_bicg, gsl_q, &omega);   // omega = dot(p'(H),q)

        if (gsl_complex_abs(omega) < EPS) {
            break;
        }

        alpha = gsl_complex_div(rho,omega);
        alpha_conj = gsl_complex_conjugate(alpha);
        neg_alpha = gsl_complex_negative(alpha);
        neg_alpha_conj = gsl_complex_negative(alpha_conj);
        
        gsl_blas_zaxpy(alpha, gsl_p, cur_gsl_x);        // x = x + alpha*p
        gsl_blas_zaxpy(neg_alpha, gsl_q, gsl_r);           // r = r - alpha*q
        gsl_blas_zaxpy(neg_alpha_conj, gsl_q_bicg, gsl_r_bicg); // r' = r' - alpha'*q'

        r_norm = gsl_blas_dznrm2(gsl_r); // norm of vector r



    }

    gsl_vector_complex_free(gsl_r);
    gsl_vector_complex_free(gsl_r_bicg);
    gsl_vector_complex_free(gsl_Ax);
    gsl_vector_complex_free(gsl_z);
    gsl_vector_complex_free(gsl_z_bicg);
    gsl_vector_complex_free(gsl_p);
    gsl_vector_complex_free(gsl_p_bicg);
    gsl_vector_complex_free(gsl_q);
    gsl_vector_complex_free(gsl_q_bicg);
    gsl_matrix_complex_free(cur_gsl_A_conjtrans);



}

gsl_vector_complex* preconditioner_solve_complex(gsl_matrix_complex *cur_gsl_A, gsl_vector_complex *gsl_r, gsl_vector_complex *gsl_z) {

    //gsl_vector *gsl_z = gsl_vector_alloc(A_dim);
    gsl_complex cur_A_element;
    gsl_complex cur_z_element;

    gsl_complex complex_1 = gsl_complex_rect(1.0, 0.0);

    for (int i=0; i<A_dim; i++) {

        cur_A_element = gsl_matrix_complex_get(cur_gsl_A, i, i);

        if (GSL_REAL(cur_A_element) == 0 && GSL_IMAG(cur_A_element) == 0) cur_A_element = complex_1;

        cur_z_element = gsl_complex_div(gsl_vector_complex_get(gsl_r, i), cur_A_element);

        gsl_vector_complex_set(gsl_z, i, cur_z_element);
    }

    return gsl_z;
}

// This function creates a conjugate vector
void make_conj_vector(gsl_vector_complex *gsl_v, gsl_vector_complex *gsl_v_conj) {

    int i;
    gsl_complex v_element;
    gsl_complex v_element_conj;

    for (i=0; i<A_dim; i++) {
        v_element = gsl_vector_complex_get(gsl_v, i);
        v_element_conj = gsl_complex_conjugate(v_element);
        gsl_vector_complex_set(gsl_v_conj, i, v_element_conj);
    }
}




