#include "ac.h"
#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "parse.h"
#include "direct_sol.h"
#include "iter_sol.h"

double complex* ac_b_vector = NULL;
gsl_vector_complex *gsl_ac_b_vector;

void ac_sweep() {

    double omega;
    
    double complex* temp_ac_A_array = (double complex*)calloc((A_dim*A_dim), sizeof(double complex));
    gsl_matrix_complex* temp_gsl_ac_A_array = gsl_matrix_complex_calloc(A_dim, A_dim);

    alloc_ac_b_vector();
    fill_ac_b_vector();

    double iter_num = 10;
    for (double freq=0; freq<iter_num; freq++) {

        omega=2*M_PI*freq;

        // fill AC array in double complex form
        fill_ac_A_array_temp(temp_ac_A_array, omega);

        // then copy in GSL form

        for (int i=0; i<A_dim; i++) {
            for (int j=0; j<A_dim; j++) {
                gsl_matrix_complex_set(temp_gsl_ac_A_array, i, j, temp_ac_A_array[i*A_dim + j]);
            }
        }
        
        solve_ac_sweep_system(temp_gsl_ac_A_array);

        // print_ac_A_array(temp_ac_A_array);
        print_gsl_matrix_complex(temp_gsl_ac_A_array, A_dim);
        memset(temp_ac_A_array, 0, sizeof(double complex) *A_dim*A_dim);




    }
}

// The b-vector in the AC analysis is always the same, so it must only be created once

void fill_ac_b_vector() {

    component* current = head;
    int m2_count_ac = 0;
    int i;

    // For each component of the circuit fill the b vector
    while (current != NULL) {
        
        if(current->comp_type == 'r'){
            fill_ac_bvector_with_r(current);
        }
        else if(current->comp_type == 'i'){
            fill_ac_bvector_with_i(current);
        }
        else if(current->comp_type == 'c'){
            fill_ac_bvector_with_c(current);
        }
        else if(current->comp_type == 'v'){
            fill_ac_bvector_with_v(current, &m2_count_ac);
        }
        else if(current->comp_type == 'l'){
            fill_ac_bvector_with_l(current, &m2_count_ac);
        }

        current = current->next;
    }

    // Also fill the gsl vector (preferably in separate function)

    gsl_ac_b_vector = gsl_vector_complex_alloc(A_dim);
    for (int i=0; i<A_dim; i++) {
        gsl_vector_complex_set(gsl_ac_b_vector, i, ac_b_vector[i]);
    }


}

// The array changes at each iteration

void fill_ac_A_array_temp(double complex* temp_ac_A_array, double omega) {

    component* current = head;
    int m2_count_ac = 0;

    // For each component of the circuit fill the arrays
    while (current != NULL) {
        
        if(current->comp_type == 'r'){
            fill_ac_A_with_r(temp_ac_A_array, current);
        }
        else if(current->comp_type == 'i'){
            fill_ac_A_with_i(temp_ac_A_array, current);
        }
        else if(current->comp_type == 'c'){
            fill_ac_A_with_c(temp_ac_A_array, current, omega);
        }
        else if(current->comp_type == 'v'){
            fill_ac_A_with_v(temp_ac_A_array, current, &m2_count_ac);
        }
        else if(current->comp_type == 'l'){
            fill_ac_A_with_l(temp_ac_A_array, current, &m2_count_ac, omega);
        }

        current = current->next;
    }
}


void alloc_ac_b_vector(){

    // Allocate memory for the array of pointers to rows
    double complex* new_vector = (double complex*)calloc((A_dim), sizeof(double complex));

    // Check if memory allocation was successful
    if (new_vector == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(0);
    }

    // Return a pointer to the created array
    ac_b_vector = new_vector;

}

void fill_ac_bvector_with_r(component *current) {

}

void fill_ac_bvector_with_i(component *current) {

    int positive_node_i;
    int negative_node_i;

    // Take the node_index of each node
    positive_node_i = find_hash_node(&node_hash_table, str_tolower(current->positive_node));
    negative_node_i = find_hash_node(&node_hash_table, str_tolower(current->negative_node));

    // If the I source is not AC, the value is 0
    if (current->spec_type != AC_SPEC) {
        return;
    }

    double complex current_i = convert_to_complex(current->spec->ac.mag, current->spec->ac.phase);

    // Case: i 0 x value
    if(positive_node_i == 0){
        ac_b_vector[negative_node_i-1] += current_i;
    }
    // Case: i x 0 value
    else if(negative_node_i == 0) {
        ac_b_vector[positive_node_i-1] -= current_i;
    }
    // Case: i x y value
    else{
        ac_b_vector[negative_node_i-1] += current_i;
        ac_b_vector[positive_node_i-1] -= current_i;
    }

    return;
}

void fill_ac_bvector_with_c(component *current) {
    
}

void fill_ac_bvector_with_v(component *current, int *m2_count_ac) {

    int positive_node_i;
    int negative_node_i;
    int m2_i = nodes_n - 1 + (*m2_count_ac);

    // Take the node_index of each node
    positive_node_i = find_hash_node(&node_hash_table, str_tolower(current->positive_node));
    negative_node_i = find_hash_node(&node_hash_table, str_tolower(current->negative_node));

    // If the I source is not AC, the value is 0
    if (current->spec_type != AC_SPEC) {
        return;
    }

    double complex current_v = convert_to_complex(current->spec->ac.mag, current->spec->ac.phase);

    ac_b_vector[m2_i] = current_v;

    (*m2_count_ac)++;
}

void fill_ac_bvector_with_l(component *current, int *m2_count_ac) {
    (*m2_count_ac)++;
}


void fill_ac_A_with_r(double complex* temp_ac_A_array, component *current) {

    int positive_node_i;
    int negative_node_i;

    double current_g = 1/current->value; // g = 1/R

    // Take the node_index of each node
    positive_node_i = find_hash_node(&node_hash_table, str_tolower(current->positive_node));
    negative_node_i = find_hash_node(&node_hash_table, str_tolower(current->negative_node));

    // Case: r 0 x value
    if (positive_node_i == 0) {
        temp_ac_A_array[(negative_node_i-1) * A_dim + (negative_node_i-1)] += current_g;
    }

    // Case: r x 0 value
    else if (negative_node_i == 0) {
        temp_ac_A_array[(positive_node_i-1) * A_dim + (positive_node_i-1)] += current_g;
    }

    // Case: x y value
    else {
        temp_ac_A_array[(negative_node_i-1) * A_dim + (negative_node_i-1)] += current_g;
        temp_ac_A_array[(positive_node_i-1) * A_dim + (positive_node_i-1)] += current_g;
        temp_ac_A_array[(negative_node_i-1) * A_dim + (positive_node_i-1)] -= current_g;
        temp_ac_A_array[(positive_node_i-1) * A_dim + (negative_node_i-1)] -= current_g;
    }
}

void fill_ac_A_with_i(double complex* temp_ac_A_array, component *current) {

}

void fill_ac_A_with_c(double complex* temp_ac_A_array, component *current, double omega) {
    int positive_node_i;
    int negative_node_i;

    double current_c = omega*current->value; // omega*c

    // Take the node_index of each node
    positive_node_i = find_hash_node(&node_hash_table, str_tolower(current->positive_node));
    negative_node_i = find_hash_node(&node_hash_table, str_tolower(current->negative_node));

    // Case: r 0 x value
    if (positive_node_i == 0) {
        temp_ac_A_array[(negative_node_i-1) * A_dim + (negative_node_i-1)] += I*current_c;
    }

    // Case: r x 0 value
    else if (negative_node_i == 0) {
        temp_ac_A_array[(positive_node_i-1) * A_dim + (positive_node_i-1)] += I*current_c;
    }

    // Case: x y value
    else {
        temp_ac_A_array[(negative_node_i-1) * A_dim + (negative_node_i-1)] += I*current_c;
        temp_ac_A_array[(positive_node_i-1) * A_dim + (positive_node_i-1)] += I*current_c;
        temp_ac_A_array[(negative_node_i-1) * A_dim + (positive_node_i-1)] -= I*current_c;
        temp_ac_A_array[(positive_node_i-1) * A_dim + (negative_node_i-1)] -= I*current_c;
    }
}

void fill_ac_A_with_v(double complex* temp_ac_A_array, component *current, int *m2_count_ac) {

    int positive_node_i;
    int negative_node_i;
    int m2_i = nodes_n - 1 + (*m2_count_ac);

    // Take the node_index of each node
    positive_node_i = find_hash_node(&node_hash_table, str_tolower(current->positive_node));
    negative_node_i = find_hash_node(&node_hash_table, str_tolower(current->negative_node));

    // Case: v 0 x value
    if(positive_node_i == 0){
        temp_ac_A_array[(negative_node_i-1)*A_dim +m2_i] = -1;
        temp_ac_A_array[m2_i*A_dim +(negative_node_i-1)] = -1;
    }
    // Case: v x 0 value
    else if(negative_node_i == 0){
        temp_ac_A_array[(positive_node_i-1)*A_dim+m2_i] = 1;
        temp_ac_A_array[m2_i*A_dim +(positive_node_i-1)] = 1;
    }
    // Case: v x y value
    else{
        temp_ac_A_array[(negative_node_i-1)*A_dim +m2_i] = -1;
        temp_ac_A_array[m2_i*A_dim +(negative_node_i-1)] = -1;
        temp_ac_A_array[(positive_node_i-1)*A_dim+m2_i] = 1;
        temp_ac_A_array[m2_i*A_dim +(positive_node_i-1)] = 1;
    }

    (*m2_count_ac)++;
}

void fill_ac_A_with_l(double complex* temp_ac_A_array, component *current, int *m2_count_ac, double omega) {

    int positive_node_i;
    int negative_node_i;
    int m2_i = nodes_n - 1 + (*m2_count_ac);

    // Take the node_index of each node
    positive_node_i = find_hash_node(&node_hash_table, str_tolower(current->positive_node));
    negative_node_i = find_hash_node(&node_hash_table, str_tolower(current->negative_node));

    // Case: l 0 x value
    if(positive_node_i == 0){
        temp_ac_A_array[(negative_node_i-1)*A_dim +m2_i] = -1;
        temp_ac_A_array[m2_i*A_dim +(negative_node_i-1)] = -1;
    }
    // Case: l x 0 value
    else if(negative_node_i == 0){
        temp_ac_A_array[(positive_node_i-1)*A_dim+m2_i] = 1;
        temp_ac_A_array[m2_i*A_dim +(positive_node_i-1)] = 1;
    }
    // Case: l x y value
    else{
        temp_ac_A_array[(negative_node_i-1)*A_dim + m2_i] = -1;
        temp_ac_A_array[m2_i*A_dim +(negative_node_i-1)] = -1;
        temp_ac_A_array[(positive_node_i-1)*A_dim + m2_i] = 1;
        temp_ac_A_array[m2_i*A_dim +(positive_node_i-1)] = 1;
    }

    // Always fill the C array (for Transient analysis)
    temp_ac_A_array[m2_i*A_dim + m2_i] = -I*omega*current->value;
 
    (*m2_count_ac)++;

    return;
}

void solve_ac_sweep_system(gsl_matrix_complex *temp_gsl_ac_A_array) {

    gsl_vector_complex *temp_gsl_x;

    temp_gsl_x = gsl_vector_complex_calloc(A_dim);

    // Solve the system based on the solver flag, generated during parsing

    if (solver_type == LU_SOL) {
        solve_LU_complex(temp_gsl_ac_A_array, gsl_ac_b_vector, temp_gsl_x);
        print_gsl_vector_complex(temp_gsl_x, A_dim);
    }


    else if (solver_type == BICG_SOL) {
        solve_bicg_complex(temp_gsl_ac_A_array, gsl_ac_b_vector, temp_gsl_x);
        print_gsl_vector_complex(temp_gsl_x, A_dim);
    }
    // else if (solver_type == SPARSE_LU_SOL) {
    //     solve_sparse_lu(sparse_cc_A, temp_gsl_b, temp_gsl_x);
    // }
    // else if (solver_type == SPARSE_BICG_SOL) {
    //     solve_sparse_bicg(sparse_cc_A, temp_gsl_b, temp_gsl_x);
    // }

    // int i;
    // int plot_node_i;
    // double b_vector_value, x_vector_value;

    // // Plot_node_i: The index of the node(s) that are to be plotted
    // // Sweep_node_i: The index of the node whose value in the b vector changes

    // // Get the values that will be printed to the files, then call add_to_plot_file
    // for (i = 0; i < plot_node_count; i++) {
    //     plot_node_i = plot_node_indexes[i];
    //     b_vector_value = cur_value;
    //     x_vector_value = gsl_vector_get(temp_gsl_x, plot_node_i);
    //     add_to_plot_file(b_vector_value, x_vector_value, i);

    // }

    // //gsl_vector_memcpy(gsl_x, temp_gsl_x); // copy the value to gsl_x, so that it is the initial value for the next cg/bicg
    // gsl_vector_free(temp_gsl_x);

}

// This function converts a complex number represented with the magnitude and phase, to its real and imaginary part
double complex convert_to_complex(double mag, double phase) {

    double phase_rad = phase*M_PI/180.0;

    double real = mag*cos(phase_rad);
    double imaginary = mag*sin(phase_rad);

    return real + I*imaginary;

}

void print_ac_b_vector() {

    int i;

    printf("ac b vector is\n");
    for (i=0; i<A_dim; i++) {
        printf("%.2lf +i*%.2lf\n", creal(ac_b_vector[i]), cimag(ac_b_vector[i]));
    }


}

void print_ac_A_array(double complex* temp_ac_A_array) {


    printf("\nac A array is\n");
    for (int i = 0; i < (A_dim); i++) {
        for (int j = 0; j < (A_dim); j++) {
            printf("%.2lf +i*%.2lf\t", creal(temp_ac_A_array[i*A_dim + j]), cimag(temp_ac_A_array[i*A_dim + j]));
        }
        printf("\n");
    }

}