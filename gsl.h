#ifndef GSL_H
#define GSL_H

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>

#define TRUE 1
#define FALSE 0

extern gsl_matrix* gsl_A;
extern gsl_vector* gsl_b;
extern gsl_vector* gsl_x;

extern int spd; // Flag that indicates if the system is SPD Or not (if Cholesky failed)

extern int *plot_node_indexes;
extern int plot_node_count;


extern FILE *filePointers[5]; 

void form_gsl_system();
void print_gsl_matrix(gsl_matrix *matrix, int dim);
void print_gsl_vector(gsl_vector *vector, int dim);
void free_gsl();
void solve_dc_sweep_system(gsl_vector *temp_gsl_b, double cur_value, char type);
void solve_dc_sweep_system_sparse(double *temp_gsl_b, double cur_value, char type);
void solve_dc_sweep_system_sparse_iter(gsl_vector *temp_gsl_b, double cur_value, char type);
void add_to_plot_file(double b_vector_value, double x_vector_value, int i);

void gslErrorHandler(const char *reason, const char *file, int line, int gsl_errno);

void add_plot_node(int node_i);
void free_plot_node();

void solve_dc_system(int solver_type);
void dc_sweep();
void dc_sweep_sparse();
void dc_sweep_sparse_iter();

#endif