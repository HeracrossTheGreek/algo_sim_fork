#ifndef PARSE_H
#define PARSE_H

#define LINE_MAX 500
#define TRUE 1
#define FALSE 0

#define NOT_FOUND -1
#define PARSING_ERROR -1
#define PARSING_SUCCESSFUL 1
#define ISNOT_BLANK 0
#define IS_BLANK 1

#define LU_SOL 0
#define CHOL_SOL 1
#define BICG_SOL 2
#define CG_SOL 3
#define SPARSE_LU_SOL 4
#define SPARSE_CHOL_SOL 5
#define SPARSE_BICG_SOL 6
#define SPARSE_CG_SOL 7

#define DELIMITERS " /t/n/r"

extern char valid_comp_list[16];
extern int nodes_n; // circuit nodes INCLUDING ground
extern int m2;      // circuit branches that correspond to voltage sources or inductors
extern int m2_i;    // Index for circuit branches of type V or L
extern int lines;   // The number of lines in the input file
extern int solver_type; // The type of solver (LU or Chol)
extern double DC_arguments[3];  // The three arguments to .dc: Low, High, Step
extern int sweep_flag; // When this flag is raised, do dc_sweep
extern float itol; // tolerance as exit condition for iterative solver
extern int nonzeros; // Number of nonzeros, used by sparse structures
extern char circuit_name[30];
extern int spd_flag;

extern int A_dim;

void number_of_lines(char *file_name);
void parse(char *file_name);
int parse_line(char *line);
int valid_comp_type(char comp_type);
int is_blank_line(const char *line);
char* str_tolower(char *str);
int is_ground(char *node);
int check_for_V_or_L(char comp_type);
int parse_plot_arg(char *token);
int parse_spice_command(char* token);
int option_command(char* token);
int dc_command(char* token);
int plot_command(char* token) ;
int increment_nonzeros(char comp_type, char* positive_node, char* negative_node);

#endif