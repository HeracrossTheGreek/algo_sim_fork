#include "direct_sol.h"
#include "iter_sol.h"
#include "gsl.h"
#include <stdio.h>
#include <string.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>
#include "mna.h"
#include "structs.h"
#include "parse.h"
#include "sparse_sol.h"
#include "time.h"
#include "transient.h"

gsl_matrix* gsl_A = NULL;
gsl_vector *gsl_b = NULL;
gsl_matrix *gsl_C = NULL;
gsl_vector *gsl_x = NULL;

FILE *filePointers[5] = {NULL};

int spd = FALSE;
int *plot_node_indexes = NULL;
int plot_node_count = 0;

// This function copies the double array A made in mna.c, to a GSL array, to be used in the linear algebra routines
// It also copies the elements of the b vector

// This function cretes the gsl arrays A and b and C, and fills them
void form_gsl_system() {

    int i,j;
    i = j = A_dim;  // A_dim = n

    gsl_A = gsl_matrix_alloc(A_dim, A_dim);
    gsl_b = gsl_vector_alloc(A_dim);
    gsl_C = gsl_matrix_alloc(A_dim, A_dim);

    // Copying each element of A and C to the respective position of the GSL array

    for (i=0; i<A_dim; i++) {
        for (j=0; j<A_dim; j++) {
            gsl_matrix_set(gsl_A, i, j, A_array[i*A_dim + j]);
            gsl_matrix_set(gsl_C, i, j, C_array[i*A_dim + j]);
        }
    }

    // Copying each element of b to the respective position of the GSL Vector

    for (i=0; i<A_dim; i++) {
        gsl_vector_set(gsl_b, i, b_array[i]);
    }
}


// This function prints the specified gsl matrix. MUST be square
void print_gsl_matrix(gsl_matrix *matrix, int dim){

    int i,j;

    printf("\n");
    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            printf ("%.2lf\t", gsl_matrix_get (matrix, i, j));
        }
        printf("\n");
    }

}

// This function prints the specified complex gsl matrix. MUST be square
void print_gsl_matrix_complex(gsl_matrix_complex *matrix, int dim){

    int i,j;

    printf("\n");
    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            gsl_complex z = gsl_matrix_complex_get(matrix, i, j);
            printf("%.2lf + i*%.2lf\t", GSL_REAL(z), GSL_IMAG(z));
        }
        printf("\n");
    }

}

// This function prints the specified gsl vector
void print_gsl_vector(gsl_vector *vector, int dim){

    int i;

    printf("\n");
    for (i = 0; i < dim; i++) {
        printf ("%.5lf\n", gsl_vector_get(vector, i));
    }
    printf("\n");

}

// This function prints the specified complex gsl vector
void print_gsl_vector_complex(gsl_vector_complex *vector, int dim){

    printf("\n");
    for (int i = 0; i < dim; i++) {
        gsl_complex z = gsl_vector_complex_get(vector, i);
        printf("%.2lf + i*%.2lf\n", GSL_REAL(z), GSL_IMAG(z));
    }
    printf("\n");

}

// This function frees all memory occupied by gsl arrays

void free_gsl() {

    gsl_vector_free(gsl_x);
    gsl_vector_free(gsl_b);
    gsl_matrix_free(gsl_A);
    gsl_matrix_free(gsl_C);
    gsl_matrix_free(gsl_LU);
    gsl_matrix_free(gsl_chol);
    gsl_permutation_free(gsl_p);
}

// Custom error handler function
void gslErrorHandler(const char *reason, const char *file, int line, int gsl_errno) {

    // Print an error message
    fprintf(stderr, "GSL Error: %s \n", reason);

}

// This function solves the linear system with the altered b vector
// Then prints the value of each plot node in a separate file, that corresponds to each different value of the sweep component

void solve_dc_sweep_system(gsl_vector *temp_gsl_b, double cur_value) {

    gsl_vector *temp_gsl_x;

    temp_gsl_x = gsl_vector_calloc(A_dim);

    //gsl_vector_memcpy(temp_gsl_x, gsl_x);   // for the iterative methods, temp_gsl_x receives the value of the previous solution

    // Solve the system based on the solver flag, generated during parsing

    if (solver_type == LU_SOL) {
        gsl_linalg_LU_solve(gsl_LU, gsl_p, temp_gsl_b, temp_gsl_x);
    }
    else if (solver_type == CHOL_SOL) {
        gsl_linalg_cholesky_solve(gsl_chol, temp_gsl_b, temp_gsl_x);
    }
    else if (solver_type == CG_SOL) {
        solve_cg(gsl_A, temp_gsl_b, temp_gsl_x);
    }
    else if (solver_type == BICG_SOL) {
        solve_bicg(gsl_A, temp_gsl_b, temp_gsl_x);
    }
    else if (solver_type == SPARSE_LU_SOL) {
        solve_sparse_lu(sparse_cc_A, temp_gsl_b, temp_gsl_x);
    }
    else if (solver_type == SPARSE_CHOL_SOL) {
        solve_sparse_chol(sparse_cc_A, temp_gsl_b, temp_gsl_x);
    }
    if (solver_type == SPARSE_CG_SOL) {
        solve_sparse_cg(sparse_cc_A, temp_gsl_b, temp_gsl_x);
    }
    else if (solver_type == SPARSE_BICG_SOL) {
        solve_sparse_bicg(sparse_cc_A, temp_gsl_b, temp_gsl_x);
    }

    int i;
    int plot_node_i;
    double b_vector_value, x_vector_value;

    // Plot_node_i: The index of the node(s) that are to be plotted
    // Sweep_node_i: The index of the node whose value in the b vector changes

    // Get the values that will be printed to the files, then call add_to_plot_file
    for (i = 0; i < plot_node_count; i++) {
        plot_node_i = plot_node_indexes[i];
        b_vector_value = cur_value;
        x_vector_value = gsl_vector_get(temp_gsl_x, plot_node_i);
        add_to_plot_file(b_vector_value, x_vector_value, i);

    }

    //gsl_vector_memcpy(gsl_x, temp_gsl_x); // copy the value to gsl_x, so that it is the initial value for the next cg/bicg
    gsl_vector_free(temp_gsl_x);

}


// Adds the node to be plotted to a list of all the nodes to be plotted
// IMPORTANT: These indexes start at 0.
void add_plot_node(int node_i) {

    if (plot_node_indexes== NULL) {
        plot_node_indexes= (int *)malloc(sizeof(int) * 1);
    } else {
        plot_node_indexes= (int *)realloc(plot_node_indexes, sizeof(int) * (plot_node_count + 1));
    }

    // Check if memory allocation failed
    if (plot_node_indexes== NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    plot_node_indexes[plot_node_count] = node_i;
    plot_node_count++;
}

// This function keeps as many file pointers as the plot commands
// Then prints the corresponding values to each one

void add_to_plot_file(double b_vector_value, double x_vector_value, int i) {

    // Check if the file corresponding to i is already open
    if (filePointers[i] == NULL) {
        char filename[60];
        snprintf(filename, sizeof(filename), "output/%s_%d.txt", circuit_name, i);

        // Open the file in write mode inside the "output" subdirectory
        filePointers[i] = fopen(filename, "w");

        // Check if the file is opened successfully
        if (filePointers[i] == NULL) {
            printf("Error opening file %d\n", i);
            return;
        }
    }

    // Write the values to the file
    fprintf(filePointers[i], "%.3e %.6e\n", b_vector_value, x_vector_value);

    // Flush the file buffer to ensure data is written immediately
    fflush(filePointers[i]);

    // Code to generate the GNU Plot command to create the plot
    char plot_command[150];
    snprintf(plot_command, sizeof(plot_command), "set terminal png; set output 'output/%s_%d.png'; plot 'output/%s_%d.txt' with lines", circuit_name, i, circuit_name, i);

    // Generate a temporary script file to run the GNU Plot commands
    FILE *gnuplotScript = fopen("plot_script.gnu", "w");
    if (gnuplotScript == NULL) {
        printf("Error creating GNU Plot script file\n");
        return;
    }

    // Write the GNU Plot command to the script file
    fprintf(gnuplotScript, "%s", plot_command);
    fclose(gnuplotScript);
    
    // // // Execute GNU Plot using the script file
    // system("gnuplot plot_script.gnu");

    // // // Clean up: remove the temporary script file
    // remove("plot_script.gnu");
}

// This function frees the memory occupied by the plot node indexes
void free_plot_node() {

    free(plot_node_indexes);
}

// This function performs DC sweep: It solves the Ax =b system for all different values of a specific voltage or current source
// The name of this source is parsed earlier and its pointer is sweep_component
void dc_sweep() {

    double cur_value;
    double low = DC_arguments[0];
    double high = DC_arguments[1];
    double step = DC_arguments[2];

    int pos_sweep_node_i, neg_sweep_node_i, sweep_node_i;

    // Temp_gsl_b: Same as gsl_b, with only the sweep value changing

    gsl_vector *temp_gsl_b = gsl_vector_alloc(A_dim);
    gsl_vector_memcpy(temp_gsl_b, gsl_b);

    component *current = sweep_component;

    // If the component is a current source, find the index of both nodes in the hash table

    // Change the current value, adding the step in each iteration, then change the bvector
    // Then solve the linear system in each iteration, in function solve_dc_sweep_system

    // Only change the b vector if the node is not ground

    if (current->comp_type == 'i') {
        pos_sweep_node_i = find_hash_node(&node_hash_table, current->positive_node) - 1; // -1 because 0 is ground
        neg_sweep_node_i = find_hash_node(&node_hash_table, current->negative_node) - 1;

        cur_value = low;
        do {
            if (pos_sweep_node_i != -1) {
                gsl_vector_set(temp_gsl_b, pos_sweep_node_i, -cur_value);
            }
            if (neg_sweep_node_i != -1) {
                gsl_vector_set(temp_gsl_b, neg_sweep_node_i, cur_value);
            }

            solve_dc_sweep_system(temp_gsl_b, cur_value); 

            cur_value = cur_value + step;
        } while (cur_value <= high);

    }

    // If the component is a voltage source, take its index in the m2 part of the b vector

    // Change the current value, adding the step in each iteration, then change the bvector
    // Then solve the linear system in each iteration, in function solve_dc_sweep_system

    if (current->comp_type == 'v') {
        sweep_node_i = nodes_n-1 +current->m2_i;

        cur_value = low;
        do {
            gsl_vector_set(temp_gsl_b, sweep_node_i, cur_value); 
            solve_dc_sweep_system(temp_gsl_b, cur_value); 
            cur_value = cur_value + step;
        } while (cur_value <= high);
    }

    gsl_vector_free(temp_gsl_b);

}

// This function performs tran sweep: It solves the Ax =b system for all different times defined
void tran_sweep() {

    gsl_vector *temp_gsl_b = gsl_vector_alloc(A_dim);
    gsl_vector *prev_gsl_x = gsl_vector_alloc(A_dim);
    gsl_vector *gsl_Cx = gsl_vector_alloc(A_dim);
    gsl_vector *curr_gsl_x = gsl_vector_alloc(A_dim);
    gsl_vector *gsl_e = gsl_vector_alloc(A_dim);
    gsl_vector *prev_temp_gsl_b = gsl_vector_alloc(A_dim);
    gsl_vector *gsl_A2x = gsl_vector_alloc(A_dim);
    gsl_matrix *gsl_A2 = gsl_matrix_alloc(A_dim, A_dim);

    gsl_vector_memcpy(temp_gsl_b, gsl_b);
    gsl_vector_memcpy(prev_gsl_x, gsl_x); // the dc solution

    // Step 1: Get the transient components and store their pointers

    component **tran_components = NULL;
    component *curr = head;
    int tran_components_size=0;

    while (curr != NULL) {
        if (curr->spec_type != NO_SPEC) {

            // Allocate memory for the new pointer
            component** newPointer = (component**) malloc(sizeof(component*));
            if (newPointer == NULL) {
                // Handle memory allocation failure
                exit(1);
            }

            // Add the pointer to the dynamic array
            tran_components = (component**) realloc(tran_components, (tran_components_size + 1) * sizeof(component*));
            if (tran_components == NULL) {
                // Handle memory reallocation failure
                exit(1);
            }

            // Add the pointer to the current component
            tran_components[tran_components_size] = curr;

            // Increment the size
            tran_components_size++;
        }

        // Move to the next component in the linked list
        curr = curr->next;
    }

    // Step 2: Create the A matrix 

    if (tran_method == BE) {

        // Step 2: Create the A matrix 

        // a) Scale array C with 1/timestep. This destroys array C!
        gsl_matrix_scale(gsl_C, 1/tran_time_step);

        // b) Create G+C/h
        gsl_matrix_add(gsl_A, gsl_C); //gsl_A == G+C/h


        // Step 3: For every time, create the b vector and call the matching solver
        double t;

        for (t=0+tran_time_step; t<=tran_fin_time; t=t+tran_time_step) {

            // Create 1/h*Cx(tk-1)

            gsl_blas_dgemv(CblasNoTrans, 1.0, gsl_C, prev_gsl_x, 0.0, gsl_Cx);

            // Create e(tk)
            create_BE_b_vector(temp_gsl_b, tran_components, tran_components_size, t);

            // b vector is e(tk) + 1/h*Cx(tk-1)
            gsl_vector_memcpy(gsl_e, temp_gsl_b);
            gsl_vector_add(gsl_e, gsl_Cx);

            // Solve the system
            solve_tran_sweep_system(gsl_e, gsl_A, curr_gsl_x, t);

            // Current x becomes previous x
            gsl_vector_memcpy(prev_gsl_x, curr_gsl_x);

        }
    }

     else { // if TR

        // Create A and some parts of b
        
        gsl_matrix_memcpy(gsl_A2, gsl_A);

        // a) Scale array C with 2/timestep. This destroys array C!
        gsl_matrix_scale(gsl_C, 2/tran_time_step);

        // b) Create G+2C/h
        gsl_matrix_add(gsl_A, gsl_C); //gsl_A == G+2C/h

        // c) Create G-2C/h
        gsl_matrix_sub(gsl_A2, gsl_C); //gsl_A2 == G-2C/h

        // d) create e(tk-1)
        gsl_vector_memcpy(prev_temp_gsl_b, temp_gsl_b);


        // Step 3: For every time, create the b vector and call the matching solver
        double t;

        for (t=0+tran_time_step; t<=tran_fin_time; t=t+tran_time_step) {

            // Create (G-2C/h)*x(tk-1)
            gsl_blas_dgemv(CblasNoTrans, 1.0, gsl_A2, prev_gsl_x, 0.0, gsl_A2x);

            // Create e(tk)
            create_BE_b_vector(temp_gsl_b, tran_components, tran_components_size, t);

            // Create e(tk-1) - (G-2C/h)*x(tk-1)
            gsl_vector_sub(prev_temp_gsl_b, gsl_A2x);

            // Create e(tk) + e(tk-1) - (G-2C/h)*x(tk-1)
            gsl_vector_memcpy(gsl_e, temp_gsl_b);
            gsl_vector_add(gsl_e, prev_temp_gsl_b);

            solve_tran_sweep_system(gsl_e, gsl_A, curr_gsl_x, t);

            // Current x becomes previous x
            gsl_vector_memcpy(prev_gsl_x, curr_gsl_x);
            // Current e becomes previous e
            gsl_vector_memcpy(prev_temp_gsl_b, temp_gsl_b);

        }

    }
}

// This function performs tran sweep for Sparse matrices: It solves the Ax =b system for all different times defined
void tran_sweep_sparse() {

    cs* sparse_tran_LHS;
    cs* sparse_tran_RHS;
    gsl_vector *temp_gsl_b = gsl_vector_alloc(A_dim);
    gsl_vector *prev_gsl_x = gsl_vector_alloc(A_dim);
    gsl_vector *gsl_Cx = gsl_vector_alloc(A_dim);
    gsl_vector *curr_gsl_x = gsl_vector_alloc(A_dim);
    gsl_vector *gsl_e = gsl_vector_alloc(A_dim);
    gsl_vector *prev_temp_gsl_b = gsl_vector_alloc(A_dim);

    gsl_vector *gsl_A2x = gsl_vector_alloc(A_dim);

    gsl_vector_memcpy(temp_gsl_b, gsl_b);
    gsl_vector_memcpy(prev_gsl_x, gsl_x); // the dc solution

    // Step 1: Get the transient components and store their pointers

    component **tran_components = NULL;
    component *curr = head;
    int tran_components_size=0;

    while (curr != NULL) {
        if (curr->spec_type != NO_SPEC) {

            // Allocate memory for the new pointer
            component** newPointer = (component**) malloc(sizeof(component*));
            if (newPointer == NULL) {
                // Handle memory allocation failure
                exit(1);
            }

            // Add the pointer to the dynamic array
            tran_components = (component**) realloc(tran_components, (tran_components_size + 1) * sizeof(component*));
            if (tran_components == NULL) {
                // Handle memory reallocation failure
                exit(1);
            }

            // Add the pointer to the current component
            tran_components[tran_components_size] = curr;

            // Increment the size
            tran_components_size++;
        }

        // Move to the next component in the linked list
        curr = curr->next;
    }
    
    if (tran_method == BE) {

        // Step 2: Create the A matrix 

        // LHS = A + C/h
        sparse_tran_LHS = cs_add(sparse_cc_A, sparse_cc_C, 1, 1/tran_time_step);


        // Step 3: For every time, create the b vector and call the matching solver
        double t;

        for (t=0+tran_time_step; t<=tran_fin_time; t=t+tran_time_step) {

            // Create 1/h*Cx(tk-1)

            gsl_vector_set_zero(gsl_Cx);
            cs_gaxpy_with_gsl_x(sparse_cc_C, prev_gsl_x, gsl_Cx);
            gsl_vector_scale(gsl_Cx, 1/tran_time_step);

            // Create e(tk)
            create_BE_b_vector(temp_gsl_b, tran_components, tran_components_size, t);

            // b vector is e(tk) + 1/h*Cx(tk-1)
            gsl_vector_memcpy(gsl_e, temp_gsl_b);
            gsl_vector_add(gsl_e, gsl_Cx);

            // Solve the system
            solve_tran_sweep_system_sparse(gsl_e, sparse_tran_LHS, curr_gsl_x, t);

            // Current x becomes previous x
            gsl_vector_memcpy(prev_gsl_x, curr_gsl_x);

        }
    }

     else {// if TR

            
        // LHS = A + 2C/h
        sparse_tran_LHS = cs_add(sparse_cc_A, sparse_cc_C, 1, 2/tran_time_step);
        sparse_tran_RHS = cs_add(sparse_cc_A, sparse_cc_C, 1, -2/tran_time_step);

        // e(tk-1)
        gsl_vector_memcpy(prev_temp_gsl_b, temp_gsl_b);

        // Step 3: For every time, create the b vector and call the matching solver
        double t;

        for (t=0+tran_time_step; t<=tran_fin_time; t=t+tran_time_step) {

            // Create (G-2C/h)*x(tk-1)

            gsl_vector_set_zero(gsl_A2x);
            cs_gaxpy_with_gsl_x(sparse_tran_RHS, prev_gsl_x, gsl_A2x);
            //gsl_vector_scale(gsl_A2x, 1/tran_time_step);

            // Create e(tk)
            create_BE_b_vector(temp_gsl_b, tran_components, tran_components_size, t);

            // Create e(tk-1) - (G-2C/h)*x(tk-1)
            gsl_vector_sub(prev_temp_gsl_b, gsl_A2x);

            // Create e(tk) + e(tk-1) - (G-2C/h)*x(tk-1)
            gsl_vector_memcpy(gsl_e, temp_gsl_b);
            gsl_vector_add(gsl_e, prev_temp_gsl_b);

            // Solve the system

            solve_tran_sweep_system_sparse(gsl_e, sparse_tran_LHS, curr_gsl_x, t);

            // Current x becomes previous x
            gsl_vector_memcpy(prev_gsl_x, curr_gsl_x);
            // Current e becomes previous e
            gsl_vector_memcpy(prev_temp_gsl_b, temp_gsl_b);

        }

    }
}

// This function solves the  DC system,
// Then writes the dc operating point in a .op file

void solve_dc_system(int solver_type) {

    clock_t t1, t2;
    double time;

    gsl_x = gsl_vector_calloc(A_dim);   // fill it with zeros, useful for the iterative methods

    t1 = clock();

    // Solve the system depending on the solver type
    if (solver_type == LU_SOL) {
        gsl_linalg_LU_solve(gsl_LU, gsl_p, gsl_b, gsl_x);
    }
    else if (solver_type == CHOL_SOL) {
        gsl_linalg_cholesky_solve(gsl_chol, gsl_b, gsl_x);
    }
    else if (solver_type == CG_SOL){
        solve_cg(gsl_A, gsl_b, gsl_x);
    }
    else if (solver_type == BICG_SOL){
        solve_bicg(gsl_A, gsl_b, gsl_x);
    }
    else if (solver_type == SPARSE_LU_SOL){
        solve_sparse_lu(sparse_cc_A, gsl_b, gsl_x);
    }
    else if (solver_type == SPARSE_CHOL_SOL){
        solve_sparse_chol(sparse_cc_A, gsl_b, gsl_x);
    }
    else if (solver_type == SPARSE_CG_SOL){
        solve_sparse_cg(sparse_cc_A, gsl_b, gsl_x);
    }
    else if (solver_type == SPARSE_BICG_SOL){
        solve_sparse_bicg(sparse_cc_A, gsl_b, gsl_x);
    }

    t2 = clock();
    time = (double) (t2 - t1) / CLOCKS_PER_SEC;
    printf("Completed in %f seconds\n", time);

    // Open the file in which the operating point will be printed
    FILE *op_file;
    char op_file_name[40];

    snprintf(op_file_name, sizeof(op_file_name), "%s%s%s", "output/", circuit_name, ".op");
    op_file = fopen(op_file_name, "w");

    // Print the solution to the file
    // First n-1 elements are the voltages of the nodes
    // Last m2 elements are the currents of the V and L components

    for (int i = 0, j=0; i < A_dim; i++) {

        if(i < nodes_n-1){
            fprintf(op_file, "%s\t\t%.5e\n", node_array[i+1], gsl_vector_get(gsl_x, i));    
        }
        else
        {
            fprintf(op_file, "%s\t\t%.5e\n", m2_array[j], gsl_vector_get(gsl_x, i));
            j++;
        }
    }

    fclose(op_file);

}

// This function converts a double to a gsl vector
void double_to_gsl(gsl_vector *gsl_v, double* v) {

    int i;

    for (i=0; i<A_dim; i++) {
        gsl_vector_set(gsl_v, i, v[i]);
    }

}

// This function converts a gsl vector to a double 
void gsl_to_double(gsl_vector *gsl_v, double* v) {

    int i;

    for (i=0; i<A_dim; i++) {
        v[i] = gsl_vector_get(gsl_v, i);

    }

}

// This function returns the gsl_vector to be used by BE method, based on the current time

void create_BE_b_vector(gsl_vector *temp_gsl_b, component **tran_components, int tran_components_size, double t) {

    component *curr = NULL;
    int sweep_node_i, pos_sweep_node_i, neg_sweep_node_i;
    double cur_value;

    // create e(tk)
    for (int i=0; i<tran_components_size; i++) {

        curr = tran_components[i];
        cur_value = get_comp_transient_val(curr, t);

        if (curr->comp_type == 'i') {
            pos_sweep_node_i = find_hash_node(&node_hash_table, curr->positive_node) - 1; // -1 because 0 is ground
            neg_sweep_node_i = find_hash_node(&node_hash_table, curr->negative_node) - 1;

            if (pos_sweep_node_i != -1) {
                gsl_vector_set(temp_gsl_b, pos_sweep_node_i, -cur_value);
            }
            if (neg_sweep_node_i != -1) {
                gsl_vector_set(temp_gsl_b, neg_sweep_node_i, cur_value);
            }
        }

        else if (curr->comp_type == 'v') {
            sweep_node_i = nodes_n-1 +curr->m2_i;
            gsl_vector_set(temp_gsl_b, sweep_node_i, cur_value); 
        }

    }


}

// Solves the tran system for the respective time, for non-sparse solvers
void solve_tran_sweep_system(gsl_vector *temp_gsl_b, gsl_matrix* gsl_A, gsl_vector *curr_gsl_x, double t) {

    // Solve the system based on the solver flag, generated during parsing

    if (solver_type == LU_SOL) {
        solve_LU_with_args(gsl_A, curr_gsl_x, temp_gsl_b);
    }
    else if (solver_type == CHOL_SOL) {
        solve_chol_with_args(gsl_A, curr_gsl_x, temp_gsl_b);
    }
    else if (solver_type == CG_SOL) {
        solve_cg(gsl_A, temp_gsl_b, curr_gsl_x);
    }
    else if (solver_type == BICG_SOL) {
        solve_bicg(gsl_A, temp_gsl_b, curr_gsl_x);
    }

    int i;
    int plot_node_i;
    double x_vector_value;

    // Plot_node_i: The index of the node(s) that are to be plotted
    // Sweep_node_i: The index of the node whose value in the b vector changes

    // Get the values that will be printed to the files, then call add_to_plot_file
    for (i = 0; i < plot_node_count; i++) {
        plot_node_i = plot_node_indexes[i];
        x_vector_value = gsl_vector_get(curr_gsl_x, plot_node_i);
        add_to_plot_file(t, x_vector_value, i);

    }

}

// Solves the tran system for the respective time, for non-sparse solvers
void solve_tran_sweep_system_sparse(gsl_vector *temp_gsl_b, cs* sparse_cc_A, gsl_vector *curr_gsl_x, double t) {

    // Solve the system based on the solver flag, generated during parsing
    
    if (solver_type == SPARSE_LU_SOL) {
        solve_sparse_lu(sparse_cc_A, temp_gsl_b, curr_gsl_x);
    }
    else if (solver_type == SPARSE_CHOL_SOL) {
        solve_sparse_chol(sparse_cc_A, temp_gsl_b, curr_gsl_x);
    }
    if (solver_type == SPARSE_CG_SOL) {
        solve_sparse_cg(sparse_cc_A, temp_gsl_b, curr_gsl_x);
    }
    else if (solver_type == SPARSE_BICG_SOL) {
        solve_sparse_bicg(sparse_cc_A, temp_gsl_b, curr_gsl_x);
    }

    int i;
    int plot_node_i;
    double x_vector_value;

    // Plot_node_i: The index of the node(s) that are to be plotted
    // Sweep_node_i: The index of the node whose value in the b vector changes

    // Get the values that will be printed to the files, then call add_to_plot_file
    for (i = 0; i < plot_node_count; i++) {
        plot_node_i = plot_node_indexes[i];
        x_vector_value = gsl_vector_get(curr_gsl_x, plot_node_i);
        add_to_plot_file(t, x_vector_value, i);

    }

}
