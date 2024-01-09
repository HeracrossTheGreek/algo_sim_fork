#include <stdio.h>
#include <stdlib.h>
#include "sparse_sol.h"
#include "mna.h"
#include "parse.h"
#include "structs.h"
#include "gsl.h"

cs *sparse_A;
cs *sparse_C;

double *b_array_sparse = NULL;
double *x_array_sparse = NULL;

int sparse_m2_count=0;
int N;

void form_sparse() { 

    component* current = head;
    int nonzero_counter = 0;
    N = nodes_n - 1 + m2;
    printf("N is %d\n", N);

    sparse_A = cs_spalloc(N, N, nonzeros,1,1);
    create_b_array_for_sparse();

    // For each component of the circuit fill the sparse array
    // Also increment the nonzero counter

    while (current != NULL) {

        if(current->comp_type == 'r'){
            sparse_fill_with_r(current, &nonzero_counter);
        }
        else if(current->comp_type == 'v'){
            sparse_fill_with_v(current, &nonzero_counter);
        }
        else if(current->comp_type == 'l'){
            sparse_fill_with_l(current, &nonzero_counter);
        }
        else if(current->comp_type == 'i'){
            sparse_fill_with_i(current);
        }

        current = current->next;
    }

    sparse_A->nz = nonzeros;

    sparse_C = cs_compress(sparse_A);

    cs_spfree(sparse_A);
    cs_dupl(sparse_C);

    // We also create gsl b for iter solver
    gsl_b = gsl_vector_alloc(N);

    for (int i=0; i<N; i++) {
        gsl_vector_set(gsl_b, i, b_array_sparse[i]);
    }

}

// Component: R ---> Fill A1*G*A1t array
void sparse_fill_with_r(component* current, int *nonzero_counter){

    int positive_node_i;
    int negative_node_i;
    int k = *nonzero_counter;


    double current_g;

    if (current->value == 0) current_g = 0;
    else current_g =  1/current->value; // g = 1/R

    // Take the node_index of each node
    positive_node_i = find_hash_node(&node_hash_table, str_tolower(current->positive_node));
    negative_node_i = find_hash_node(&node_hash_table, str_tolower(current->negative_node));

    // Case: r 0 x value
    if (positive_node_i == 0) {

        add_to_sparse_A(k, negative_node_i-1, negative_node_i-1, current_g);
        *nonzero_counter += 1;
    }

    // Case: r x 0 value
    else if (negative_node_i == 0) {

        add_to_sparse_A(k, positive_node_i-1, positive_node_i-1, current_g);
        *nonzero_counter += 1;
    }

    // Case: x y value
    else {
        add_to_sparse_A(k, negative_node_i-1, negative_node_i-1, current_g);
        add_to_sparse_A(k+1, positive_node_i-1, positive_node_i-1, current_g);
        add_to_sparse_A(k+2, negative_node_i-1, positive_node_i-1, -current_g);
        add_to_sparse_A(k+3, positive_node_i-1, negative_node_i-1, -current_g);
        *nonzero_counter += 4;
    }

    // return;
}

// Component: I ---> Fill b array
void sparse_fill_with_i(component* current){

    int positive_node_i;
    int negative_node_i;

    double current_i = current->value;

    // Take the node_index of each node
    positive_node_i = find_hash_node(&node_hash_table, str_tolower(current->positive_node));
    negative_node_i = find_hash_node(&node_hash_table, str_tolower(current->negative_node));

    // Case: i 0 x value
    if(positive_node_i == 0){
        b_array_sparse[negative_node_i-1] += current_i;
    }
    // Case: i x 0 value
    else if(negative_node_i == 0) {
        b_array_sparse[positive_node_i-1] -= current_i;
    }
    // Case: i x y value
    else{
        b_array_sparse[negative_node_i-1] += current_i;
        b_array_sparse[positive_node_i-1] -= current_i;
    }

    return;
}

// Component: V ---> Fill A2t, A2 array and b array
void sparse_fill_with_v(component* current, int *nonzero_counter) {

    int positive_node_i;
    int negative_node_i;
    int sparse_m2_i = nodes_n - 1 + sparse_m2_count;
    int k = *nonzero_counter;

    // Take the node_index of each node
    positive_node_i = find_hash_node(&node_hash_table, str_tolower(current->positive_node));
    negative_node_i = find_hash_node(&node_hash_table, str_tolower(current->negative_node));

    // Case: v 0 x value
    if(positive_node_i == 0){
        add_to_sparse_A(k, negative_node_i-1, sparse_m2_i, -1);
        add_to_sparse_A(k+1, sparse_m2_i, negative_node_i-1, -1);
        *nonzero_counter += 2;
    }
    // Case: v x 0 value
    else if(negative_node_i == 0){
        add_to_sparse_A(k, positive_node_i-1, sparse_m2_i, 1);
        add_to_sparse_A(k+1, sparse_m2_i, positive_node_i-1, 1);
        *nonzero_counter += 2;
    }
    // Case: v x y value
    else{
        add_to_sparse_A(k, negative_node_i-1, sparse_m2_i, -1);
        add_to_sparse_A(k+1, sparse_m2_i, negative_node_i-1, -1);
        add_to_sparse_A(k+2, positive_node_i-1, sparse_m2_i, 1);
        add_to_sparse_A(k+3, sparse_m2_i, positive_node_i-1, 1);
        *nonzero_counter += 4;
    }

    b_array_sparse[sparse_m2_i] = current->value;

    sparse_m2_count++;
}

// Component: L ---> Fill A2t, A2 array and b array
void sparse_fill_with_l(component* current, int *nonzero_counter) {

    int positive_node_i;
    int negative_node_i;
    int sparse_m2_i = nodes_n - 1 + sparse_m2_count;
    int k = *nonzero_counter;

    // Take the node_index of each node
    positive_node_i = find_hash_node(&node_hash_table, str_tolower(current->positive_node));
    negative_node_i = find_hash_node(&node_hash_table, str_tolower(current->negative_node));

    // Case: v 0 x value
    if(positive_node_i == 0){
        add_to_sparse_A(k, negative_node_i-1, sparse_m2_i, -1);
        add_to_sparse_A(k+1, sparse_m2_i, negative_node_i-1, -1);
        *nonzero_counter += 2;
    }
    // Case: v x 0 value
    else if(negative_node_i == 0){
        add_to_sparse_A(k, positive_node_i-1, sparse_m2_i, 1);
        add_to_sparse_A(k+1, sparse_m2_i, positive_node_i-1, 1);
        *nonzero_counter += 2;
    }
    // Case: v x y value
    else{
        add_to_sparse_A(k, negative_node_i-1, sparse_m2_i, -1);
        add_to_sparse_A(k+1, sparse_m2_i, negative_node_i-1, -1);
        add_to_sparse_A(k+2, positive_node_i-1, sparse_m2_i, 1);
        add_to_sparse_A(k+3, sparse_m2_i, positive_node_i-1, 1);
        *nonzero_counter += 4;
    }

    b_array_sparse[sparse_m2_i] = 0;

    sparse_m2_count++;
}

void add_to_sparse_A(int k, int i, int j, double x) {
        sparse_A->i[k] = i;
        sparse_A->p[k] = j;
        sparse_A->x[k] = x;
}

void create_b_array_for_sparse() {

    // Allocate memory for the array of pointers to rows
    b_array_sparse = (double*)calloc(N, sizeof(double));
    x_array_sparse = (double*)calloc(N, sizeof(double));

    if(x_array_sparse == NULL){
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Check if memory allocation was successful
    if (b_array_sparse == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
}

void print_sparse_arrays() {

    printf("\n *** Printing A array *** \n");

    cs_print(sparse_C, 0);

    printf("\n *** Printing b array *** \n");

    for (int k = 0; k < N; k++) {
        printf("%.2lf\t", b_array_sparse[k]);
    }

    printf("\n *** Printing x array *** \n");

    for (int l = 0; l < N; l++) {
        printf("%.2lf\t", x_array_sparse[l]);
    }

    printf("\n");
}

void print_sparse_x() {

    for (int l = 0; l < N; l++) {
        printf("%.2lf\t", x_array_sparse[l]);
    }

    printf("\n");
}