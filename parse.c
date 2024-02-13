#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "parse.h"
#include "gsl.h"
#include "direct_sol.h"

// The valid component types //
// Transistor component types are parseable but not used //

char valid_comp_list[16] = {'V', 'I', 'R', 'C', 'L', 'D', 'M', 'Q', 
                           'v', 'i', 'r', 'c', 'l', 'd', 'm', 'q'};

int nodes_n = 0;
int m2 = 0;
int m2_i = 0;
int hash_table_size = 0;
int lines = 0;
int dc_sweep_flag = 0;
int tran_sweep_flag = 0;
int solver_type = LU_SOL;
double DC_arguments[3];
float itol = 1e-3;
int nonzeros_A = 0;
int nonzeros_C = 0;
int spd_flag_circuit = 1;
double tran_time_step = 0;
double tran_fin_time = 0;
int tran_method = TR;

char circuit_name[30];  // the name circuit taken from the file name

// This function counts the lines of the file //
void number_of_lines(char* file_name){

    FILE * input_file;
    int character;

    input_file = fopen(file_name, "r");

    // Check if the file is valid
    if (input_file == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    while(!feof(input_file)){
        character = fgetc(input_file);
        if(character == '\n'){
            lines++;
        }
    }

    fclose(input_file);
    return;
}

// This function parses the input spice file
void parse(char* file_name) {

    FILE *input_file;
    char line[LINE_MAX];
    int parse_line_result;

    // Find the number of lines in the file to allocate the hashtable

    number_of_lines(file_name);
    create_hash_table(lines/2);

    input_file = fopen(file_name, "r");

    // Check if the file is valid
    if (input_file == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    // Register the circuit name based on the file name
    strcpy(circuit_name, strtok(file_name+6, "."));
    printf("Circuit name is %s\n", circuit_name);

    // Parse every line until end of file
    while (fgets(line, LINE_MAX, input_file) != NULL) {

        if (is_blank_line(line) == IS_BLANK) {
            continue;
        }
        parse_line_result = parse_line(line);

        // If the parsing fails, free the memory and exit.
        if (parse_line_result == PARSING_ERROR) {
            printf("Invalid Syntax!! at line %s\n", line);
            free_comp_list(head);
            free_hash_table(&node_hash_table);
            fclose(input_file);
            exit(1);
        }
    }

    fclose(input_file);
}

int parse_line(char* line) {

    char *token = NULL;
    char comp_type;
    char* comp_name = NULL;
    char* positive_node = NULL;
    char *negative_node = NULL;
    double value;
    int node_found;

    transient_spec_type spec_type = NO_SPEC; //default
    transient_spec* my_spec = NULL;


    // String 1: Type of circuit element & name
    token = strtok(line, " \t\n\r");

    // If the first character is an asterisk, the line is a comment
    if (token[0] == '*'){
        return 1; 
    }

    // Spice command
    if (token[0] == '.') {

        int parse_result = parse_spice_command(token);

        if (parse_result == PARSING_ERROR) {
            return PARSING_ERROR;
        }

        return PARSING_SUCCESSFUL;
    }

    if ((token == NULL) || (valid_comp_type(token[0]) == FALSE)) {
        return PARSING_ERROR;
    }

    // Store the component type and name in lowercase
    comp_type = tolower(token[0]);
    comp_name = str_tolower(token);

    // If the component is V or L, we must add it to the list of m2 components, and increment the m2 counter
    // An index number is assigned for the respective component struct field

    if ((check_for_V_or_L(comp_type)) == TRUE) {
        spd_flag_circuit = 0;   // a flag that disables SPD methods
        m2_i = m2;
        add_m2_array(comp_name);
        m2++;
    }
    else {
        m2_i = -1;  // If it is not V or L, it has no m2 index
    }



    // String 2: Positive node name
    token = strtok(NULL, " \t\n\r");
    if (token == NULL) {
        return PARSING_ERROR;
    }

    positive_node = token;

    // Search  for the node in the hash table. If it doesnt exist, add it in lowercase

    node_found = find_hash_node(&node_hash_table, str_tolower(positive_node));

    if (node_found == NOT_FOUND) {
        insert_node(&node_hash_table, str_tolower(positive_node));

        // If the node is not found, we also add it to the node array
        // The node array is something like a reverse hash table
        // It contains all the nodes in parsing order.

        add_node_array(str_tolower(positive_node));

        nodes_n++;
    }  

    // String 3: Negative node name
    token = strtok(NULL, " \t\n\r");
    if (token == NULL) {
        return PARSING_ERROR;
    }

    negative_node = token;

    // Search  for the node in the hash table. If it doesnt exist, add it in lowercase

    node_found = find_hash_node(&node_hash_table, str_tolower(negative_node));

    if (node_found == NOT_FOUND) {
        insert_node(&node_hash_table, str_tolower(negative_node));
                add_node_array(str_tolower(negative_node));

        nodes_n++;
    } 
 
    // String 4: Numeric value
    token = strtok(NULL, " \t\r\n");
    if (token == NULL) {
        return PARSING_ERROR;
    }

    // strtod converts the string to a float
    value = strtod(token, NULL);

    // String 5: (optional) Parse the type of math function (waveform), or AC mag and phase
    token = strtok(NULL, " \t\r\n(");
    if (token != NULL) {
        if (comp_type != 'v' && comp_type != 'i') {
            printf("Error! Can only have transient specs /AC at V or I component!\n");
            return PARSING_ERROR;
        }

        // Parse the spec type, and check if it is valid
        spec_type = parse_spec_type(str_tolower(token));

        if (spec_type == PARSING_ERROR) {
            printf("Invalid spec type!\n");
            return PARSING_ERROR;
        }


        // Depending on the spec type, parse a specific number of arguments

        if (spec_type == EXP_SPEC) {
            exp_spec *my_exp_spec;
            my_exp_spec = (exp_spec*) malloc(sizeof(exp_spec));

            char *tokens[6];

            for (int i=0; i<6; i++) {
                tokens[i] = strtok(NULL, " \t\r\n");
                if (tokens[i] == NULL) {
                    return PARSING_ERROR;
                }
            }

            my_exp_spec->i1 = strtod(tokens[0]+1, NULL);    // + 1 to ignore the parenthesis at [0]
            my_exp_spec->i2 = strtod(tokens[1], NULL);
            my_exp_spec->td1 = strtod(tokens[2], NULL);
            my_exp_spec->tc1 = strtod(tokens[3], NULL);
            my_exp_spec->td2 = strtod(tokens[4], NULL);
            my_exp_spec->tc2 = strtod(tokens[5], NULL);

            my_spec = (transient_spec *) my_exp_spec;

        }

        else if (spec_type == SIN_SPEC) {
            sin_spec *my_sin_spec;
            my_sin_spec = (sin_spec*) malloc(sizeof(sin_spec));

            char *tokens[6];

            for (int i=0; i<6; i++) {
                tokens[i] = strtok(NULL, " \t\r\n");
                if (tokens[i] == NULL) {
                    return PARSING_ERROR;
                }
            }

            my_sin_spec->i1 = strtod(tokens[0]+1, NULL);    // + 1 to ignore the parenthesis at [0]
            my_sin_spec->ia = strtod(tokens[1], NULL);
            my_sin_spec->fr = strtod(tokens[2], NULL);
            my_sin_spec->td = strtod(tokens[3], NULL);
            my_sin_spec->df = strtod(tokens[4], NULL);
            my_sin_spec->ph = strtod(tokens[5], NULL);

            my_spec = (transient_spec *) my_sin_spec;

        }

        else if (spec_type == PULSE_SPEC) {
            pulse_spec *my_pulse_spec;
            my_pulse_spec = (pulse_spec*) malloc(sizeof(pulse_spec));

            char *tokens[7];

            for (int i=0; i<7; i++) {
                tokens[i] = strtok(NULL, " \t\r\n");
                if (tokens[i] == NULL) {
                    return PARSING_ERROR;
                }
            }

            my_pulse_spec->i1 = strtod(tokens[0]+1, NULL);    // + 1 to ignore the parenthesis at [0]
            my_pulse_spec->i2 = strtod(tokens[1], NULL);
            my_pulse_spec->td = strtod(tokens[2], NULL);
            my_pulse_spec->tr = strtod(tokens[3], NULL);
            my_pulse_spec->tf = strtod(tokens[4], NULL);
            my_pulse_spec->pw = strtod(tokens[5], NULL);
            my_pulse_spec->per = strtod(tokens[6], NULL);

            my_spec = (transient_spec *) my_pulse_spec;

        }

        // PWL always comes at pairs

        else if (spec_type == PWL_SPEC) {
            pwl_spec *my_pwl_spec;
            my_pwl_spec = (pwl_spec*) malloc(sizeof(pwl_spec));
            my_pwl_spec->pairs = 0;
            my_pwl_spec->i = NULL;
            my_pwl_spec->t = NULL;

            char *token_t;
            char *token_i;

            while(1) {

                // t
                token_t = strtok(NULL, " \t\r\n");
                if (token_t == NULL) {
                    break;
                }

                // i
                token_i = strtok(NULL, " \t\r\n");
                if (token_i == NULL) {
                    return PARSING_ERROR;
                }

                my_pwl_spec->pairs++;
                my_pwl_spec->t = realloc(my_pwl_spec->t, my_pwl_spec->pairs * sizeof(double));
                my_pwl_spec->i = realloc(my_pwl_spec->i, my_pwl_spec->pairs * sizeof(double));

                my_pwl_spec->t[my_pwl_spec->pairs-1] = strtod(token_t+1, NULL);
                my_pwl_spec->i[my_pwl_spec->pairs-1] = strtod(token_i, NULL);


            }

            my_spec = (transient_spec *) my_pwl_spec;

        }

        else if (AC_SPEC) {

            ac_spec *my_ac_spec;
            my_ac_spec = (ac_spec*) malloc(sizeof(ac_spec));

            char *tokens[2];

            for (int i=0; i<2; i++) {
                tokens[i] = strtok(NULL, " \t\r\n");
                if (tokens[i] == NULL) {
                    return PARSING_ERROR;
                }
            }

            my_ac_spec->mag = strtod(tokens[0], NULL);    // + 1 to ignore the parenthesis at [0]
            my_ac_spec->phase = strtod(tokens[1], NULL);

            my_spec = (transient_spec *) my_ac_spec;

        }

        
    }

    // Add the component to the linked list
    append_component(&head, &tail, comp_type, comp_name, positive_node, negative_node, value, spec_type, my_spec);

    nonzeros_A += increment_nonzeros_A(comp_type, positive_node, negative_node);    // count how many nonzeros should be added in A (for sparse methods)
    nonzeros_C += increment_nonzeros_C(comp_type, positive_node, negative_node);    // count how many nonzeros should be added in C (for sparse methods)

    return PARSING_SUCCESSFUL;

}

// This function checks if the component type is valid, i.e. if it starts with one of the 8 key letters
int valid_comp_type(char comp_type) {

    int i;
    for (i = 0; i < 16; i++) {
        if (comp_type == valid_comp_list[i]) {
            return TRUE;
        }
    }

    return FALSE;
}

// This function checks if the line is blank
int is_blank_line(const char *line) {

    for (int i = 0; line[i] != '\0'; i++) {
        if (!isspace(line[i])) {
            return ISNOT_BLANK;
        }
    }
    return IS_BLANK;
}

// This function turns a string to lowercase
char* str_tolower(char *str) {

    for (int i = 0; str[i] != '\0'; i++) {
        str[i] = tolower(str[i]);
    }

    return str;
}

// This function checks if the given node is ground or not
int is_ground(char *node) {

    if ((node[0] == '0') && (node[1] == '\0')) {
        return 1;
    }

    return 0;
}

// This function checks if the component type is a voltage source or an inductor
int check_for_V_or_L(char comp_type) {

    if (comp_type == 'v' || comp_type == 'l') {
        return 1;
    }

    return 0;

}

// This function checks if the arguments for .PLOT or .PRINT are valid. 
// It returns the node index of the node, if it is valid.

int parse_plot_arg(char *token) {
    
    int node_i;

    // The argument must be v(<node>) or i(<node)

    if ((tolower(token[0]) != 'v') && (tolower(token[0] != 'i'))) {
        return PARSING_ERROR;
    }
    if (token[1] != '(') {
        return PARSING_ERROR;
    }

    int token_len = strlen(token);

    if (token[token_len - 1] != ')') {
        return PARSING_ERROR;
    }

    char *plot_node_str = (char *) malloc(sizeof(char) * (token_len - 3) + 1); // -2 for the parentheses and -1 for v or i and +1 for \0
    strncpy(plot_node_str, token + 2, token_len-3);
    plot_node_str[token_len-3] = '\0';

    // Searches if the node exists, in the hash table
    node_i = find_hash_node(&node_hash_table, plot_node_str);

    if (node_i == NOT_FOUND) {
        printf("Error! The node does not exist.\n");
        return PARSING_ERROR;
    }

    free(plot_node_str);

    return node_i;

}

// This function parses all spice commands (ending with .)
int parse_spice_command(char* token)
{

    int parsing_result = 0;
    // .options //
    if (strcmp(str_tolower(token), ".options") == 0)
    {
        parsing_result = option_command(token);
    }

    // .dc //
    else if (strcmp(str_tolower(token), ".dc") == 0)
    {
        parsing_result = dc_command(token);
    }

    // .plot or .print //
    else if ((strcmp(str_tolower(token), ".plot") == 0) || (strcmp(str_tolower(token), ".print") == 0))
    {
        parsing_result = plot_command(token);
    }

    //.tran //
    else if (strcmp(str_tolower(token), ".tran") == 0) {
        parsing_result = tran_command(token);

    }

    return parsing_result; //SUCCESSFUL or ERROR
}


int option_command(char* token) {

    int spd_flag = 0;
    int iter_flag = 0;
    int sparse_flag = 0;

    // Parsing of the OPTIONS command ends after the full command is parsed
    while(1) {
        token = strtok(NULL, " \t\n\r");
        if(token == NULL) {
            break;
        }

        else if (strcmp(str_tolower(token), "spd") == 0) {
            spd_flag = 1;
            if (spd_flag_circuit == 0) {
                printf("Error! System is not SPD!\n");
                return PARSING_ERROR;
            }
        }

        else if (strcmp(str_tolower(token), "iter") == 0) {
            iter_flag = 1;
        }

        else if (strcmp(str_tolower(token), "sparse") == 0) {
            sparse_flag = 1;
        }
        else if (strncmp(str_tolower(token), "itol=", strlen("itol=")) == 0) {
            itol = strtof(token + strlen("itol="), NULL);
        }
        else if (strcmp(str_tolower(token), "method=be") == 0) {
            tran_method = BE;
        }
    }

    // solver_type has values from 0 to 7, so binary arithmetic can be used

    solver_type = 4*sparse_flag + 2*iter_flag + spd_flag;

    // if      (sparse_flag == 0 && iter_flag == 0 && spd_flag == 0) solver_type = LU_SOL;
    // else if (sparse_flag == 0 && iter_flag == 0 && spd_flag == 1) solver_type = CHOL_SOL;
    // else if (sparse_flag == 0 && iter_flag == 1 && spd_flag == 0) solver_type = BICG_SOL;
    // else if (sparse_flag == 0 && iter_flag == 1 && spd_flag == 1) solver_type = CG_SOL;
    // else if (sparse_flag == 1 && iter_flag == 0 && spd_flag == 0) solver_type = SPARSE_LU_SOL;
    // else if (sparse_flag == 1 && iter_flag == 0 && spd_flag == 1) solver_type = SPARSE_CHOL_SOL;
    // else if (sparse_flag == 1 && iter_flag == 1 && spd_flag == 0) solver_type = SPARSE_BICG_SOL;
    // else if (sparse_flag == 1 && iter_flag == 1 && spd_flag == 1) solver_type = SPARSE_CG_SOL;



    return PARSING_SUCCESSFUL;
}

// This function parses the .dc commands and registers its arguments

int dc_command(char* token){

    token = strtok(NULL, " \t\n\r");
    if (token == NULL)
    {
        return PARSING_ERROR;
    }

    int comp_exists = find_component(str_tolower(token));

    if (comp_exists == NOT_FOUND)
    {
        printf("Error! .DC Argument is not an existing component.\n");
        return PARSING_ERROR;
    }
    else if (comp_exists == NOT_V_OR_I)
    {
        printf("Error! .DC argument is not a voltage or current source.\n");
        return PARSING_ERROR;
    }

    for (int i = 0; i < 3; i++)
    {
        token = strtok(NULL, " \t\r\n");
        if (token == NULL)
        {
            return PARSING_ERROR;
        }
        else
        {
            DC_arguments[i] = atof(token);
        }
    }
    dc_sweep_flag = 1;
    return PARSING_SUCCESSFUL;
}

// This function parses the .plot command
int plot_command(char* token) {

    token = strtok(NULL, " \t\n\r");
    if (token == NULL)
    {
        return PARSING_ERROR;
    }

    int node_i = parse_plot_arg(token);
    if (node_i == PARSING_ERROR)
    {
        return PARSING_ERROR;
    }

    add_plot_node(node_i-1);

    return PARSING_SUCCESSFUL;
}

// This function returns the number of nonzeros to be added in A for the specified component
int increment_nonzeros_A(char comp_type, char* positive_node, char* negative_node) {

    int node_is_ground = FALSE;

    if (is_ground(positive_node) || is_ground(negative_node)) {
        node_is_ground = TRUE;
    }

    if (comp_type == 'v' || comp_type == 'l') {
        if (node_is_ground) {
            return 2;
        }
        else {
            return 4;
        }
    }
    else if (comp_type == 'r') {
        if (node_is_ground) {
            return 1;
        }
        else {
            return 4;
        }
    }

    // If component is I or C, no element is added, so no nonzeros
    return 0;

}

// This function returns the number of nonzeros to be added in C (transient) for the specified component
int increment_nonzeros_C(char comp_type, char* positive_node, char* negative_node) {

    int node_is_ground = FALSE;

    if (is_ground(positive_node) || is_ground(negative_node)) {
        node_is_ground = TRUE;
    }

    if (comp_type == 'c') {
        if (node_is_ground) {
            return 1;
        }
        else {
            return 4;
        }
    }
    else if (comp_type == 'l') {
        return 1;
    }

    // If component is V or R or I, no element is added in C, so no nonzeros
    return 0;

}

// This function parses the type of transient spec. 
// Returns PARSING_ERROR if not valid, or the type of spec if valid
transient_spec_type parse_spec_type(char *token) {

    if      (strcmp(token, "exp")   == 0) return EXP_SPEC;
    else if (strcmp(token, "sin")   == 0) return SIN_SPEC;
    else if (strcmp(token, "pulse") == 0) return PULSE_SPEC;
    else if (strcmp(token, "pwl")   == 0) return PWL_SPEC;
    else if (strcmp(token, "ac")    == 0) return AC_SPEC;

    // else if none of that
    return PARSING_ERROR;

}

// This function parses the .tran command
int tran_command(char *token) {

    token = strtok(NULL, " \t\n\r");
    if (token == NULL)
    {
        return PARSING_ERROR;
    }

    tran_time_step = strtod(token, NULL);

    token = strtok(NULL, " \t\n\r");
    if (token == NULL)
    {
        return PARSING_ERROR;
    }

    tran_fin_time = strtod(token, NULL);

    tran_sweep_flag = 1;

    return PARSING_SUCCESSFUL;
}

