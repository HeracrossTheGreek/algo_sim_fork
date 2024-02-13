#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "main.h"
#include "parse.h"
#include "structs.h"
#include "mna.h"
#include "direct_sol.h"
#include "gsl.h"
#include "iter_sol.h"
#include "sparse_sol.h"
#include "ac.h"

#include "complex.h"

int A_dim;

int main(int argc, char* argv[]) {

    char file_path[40];

    double complex test = 2 + 3*I;
    // If there is no argument (or multiple arguments) exit
    if (argc!=2) {
        printf("Error! No file specified. Use ./spice_sim <file_name>\n");
        return 0;
    }
    
    strcpy(file_path, argv[1]);

    if (strncmp(file_path, "input/", 6) != 0) {
        printf("Error! The input file must be in directory input!\n");
        return 0;
    }

    parse(file_path);
    A_dim = nodes_n-1 + m2;    // A: (n-1+m2)(n-1+m2)

    printf("Solver type is %d\n", solver_type);
    switch (solver_type) {
        case LU_SOL:
            create_equations();
            form_gsl_system();
            form_LU();
            solve_dc_system(LU_SOL);
            break;

        case CHOL_SOL:
            create_equations();
            form_gsl_system();
            form_chol();
            if (spd == TRUE) {
                solve_dc_system(CHOL_SOL);
            }
            break;

        case CG_SOL:
            create_equations();
            form_gsl_system();
            solve_dc_system(CG_SOL);
            break;

        case BICG_SOL:
            create_equations();
            form_gsl_system();
            solve_dc_system(BICG_SOL);
            break;

        case SPARSE_LU_SOL:
            form_sparse();
            printf("Sparse system formed\n");
            solve_dc_system(SPARSE_LU_SOL);
            break;

        case SPARSE_CHOL_SOL:
            form_sparse();
            printf("Sparse system formed\n");
            solve_dc_system(SPARSE_CHOL_SOL);
            break;

        case SPARSE_CG_SOL:
            form_sparse();
            printf("Sparse system formed\n");
            solve_dc_system(SPARSE_CG_SOL);
            break;

        case SPARSE_BICG_SOL:
            form_sparse();
            printf("Sparse system formed\n");
            solve_dc_system(SPARSE_BICG_SOL);
            break;

        default:
            // Handle any other cases if needed
            break;
    }

    if (dc_sweep_flag == TRUE) {
        dc_sweep();
    }
    if (tran_sweep_flag == TRUE) {
        if (solver_type<4) tran_sweep();
        else tran_sweep_sparse();
    }

    int ac_sweep_flag = TRUE;
    if (ac_sweep_flag == TRUE) {
        ac_sweep();
    }
    

    print_ac_b_vector();
    print_gsl_vector_complex(gsl_ac_b_vector, A_dim);
    
    // Close all open files before exiting the program
    for (int i = 0; i < 5; i++) {
        if (filePointers[i] != NULL) {
            fclose(filePointers[i]);
            filePointers[i] = NULL;
        }
    }
    printf("The program is finished!\n");

    free_gsl();
    free_hash_table(&node_hash_table);
    free_comp_list(head);
    free_A_array();
    free_b_array();
    free_node_array();
    free_m2_array();
    free_plot_node();
    free(b_array_sparse);
    free(x_array_sparse);
    cs_spfree(sparse_cc_A);   
    
    return 0;

}