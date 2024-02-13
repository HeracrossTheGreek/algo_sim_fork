#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "structs.h"
#include "parse.h"

component* head = NULL;
component* tail = NULL;
component *sweep_component = NULL;

int index_counter = 1;
hash_table node_hash_table;

char** node_array;
int node_array_index_counter = 0;
char** m2_array;


// ------- Component Structure ------ //


// This function creates a new component
component* create_component(char comp_type, const char* comp_name, const char* positive_node, const char* negative_node, double value, transient_spec_type spec_type, transient_spec* spec) {

    component* new_node = (component*) malloc(sizeof(component));
    if (new_node == NULL) {
        printf("Error! Not enough memory.\n");
        exit(1);
    }

    new_node->comp_type = comp_type;

    new_node->comp_name = (char*) malloc(strlen(comp_name) + 1);
    new_node->positive_node = (char*) malloc(strlen(positive_node) + 1);
    new_node->negative_node = (char*) malloc(strlen(negative_node) + 1);

    if (new_node->comp_name == NULL || new_node->positive_node == NULL || new_node->negative_node == NULL) {
        printf("Error! Not enough memory.\n");
        exit(1);
    }

    strcpy(new_node->comp_name, comp_name);
    strcpy(new_node->positive_node, positive_node);
    strcpy(new_node->negative_node, negative_node);
    new_node->value = value;
    new_node->m2_i = m2_i;
    new_node->spec_type = spec_type;
    new_node->spec = spec;
    new_node->next = NULL;

    return new_node;
}

// This functions appends a component to the end of the list
void append_component(component** head, component **tail, char comp_type, const char* comp_name, const char* positive_node, const char* negative_node, double value, transient_spec_type spec_type, transient_spec* spec) {

    component* new_node = create_component(comp_type, comp_name, positive_node, negative_node, value, spec_type, spec);

    // If the head is NULL, the lsit is empty, so we add it in both ends. Else, we only update the tail
	if (*head == NULL) {
        *head = new_node;
        *tail = new_node;
    } else {
        (*tail)->next = new_node;
        *tail = new_node;
    }
}

// This functions takes the name of a component and searches it in the list
int find_component(const char* comp_name) {

    component* current = head;

    while (current !=NULL) {
        if (strcmp(current->comp_name, comp_name) == 0) {
            break;
        }
        current = current->next;
    }

    if (current == NULL) {
        return NOT_FOUND;
    }
    else if ((tolower(current->comp_type) != 'v') && (tolower(current->comp_type) != 'i')) {
        return NOT_V_OR_I;
    }
    else {
        sweep_component = current;
        return FOUND;
    }
}

// This function prints all the components in the list
void print_comp_list(component* head) {

    component* current = head;

    while (current != NULL) {
        printf("Type: %c, Name: %s, Pos Node: %s, Neg Node: %s, Value: %lf, Spec Type: %d, m2_i: %d\n",
               current->comp_type, current->comp_name, current->positive_node, current->negative_node, current->value, current->spec_type, current->m2_i);
        print_spec_numbers(current->spec_type, current->spec);
        current = current->next;
    }
}

// This function frees the memory used in the list
void free_comp_list(component* head) {

    component* current = head;

    while (current != NULL) {
        component* temp = current;
        current = current->next;
        free(temp->comp_name);
        free(temp->positive_node);
        free(temp->negative_node);
        if (temp->spec_type == PWL_SPEC) {
            free(temp->spec->pwl.t);
            free(temp->spec->pwl.i);
        }
        free(temp->spec);
        free(temp);
    }
}


// ------- Hash Table Structure ------ //

// djb2 hash function
unsigned long hash(char *str) {
    int hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return (hash);
}

// This function creates the hash table dynamically //
void create_hash_table(int size) {

        node_hash_table.size = size;
        node_hash_table.table = (hash_node **) malloc(size * sizeof(hash_node *));
        if (node_hash_table.table == NULL) {
            printf("Error! Not enough memory.\n");
            exit(EXIT_FAILURE);
        }

        // Initialize the table with NULL
        for (int i = 0; i < size; i++) {
            node_hash_table.table[i] = NULL;
        }
}


// This function creates a new hash node.
hash_node *create_hash_node(char *node_str) {
    hash_node *new_node = (hash_node *) malloc(sizeof(hash_node));
    if (new_node == NULL) {
        printf("Eror! Not enough memory.\n");
        exit(EXIT_FAILURE);
    }

    new_node->node_str = malloc(strlen(node_str) + 1);
    strcpy(new_node->node_str, node_str);

    // Node index: The order in which each node is added. Irrelevant to the hash key.
    // Reserve the node index value of "0" for the ground node
    if(strcmp(new_node->node_str,"0")==0){
        new_node->node_index = 0;
        new_node->next = NULL;

        return new_node;    
    }
    new_node->node_index = index_counter;

    new_node->next = NULL;
    index_counter++;

    return new_node;
}

// This function inserts a string into the hash table
void insert_node(hash_table *ht, char *node_str) {
    
    int index = hash((char *)node_str)%node_hash_table.size;

    hash_node *new_node = create_hash_node(node_str);

    // Insert the node at the beginning of the linked list (separate chaining)
    new_node->next = ht->table[index];
    ht->table[index] = new_node;
}

// This function checks if a string is in the hash table
int find_hash_node(hash_table *ht, char *node_str) {

    int index = hash((char *)node_str)%node_hash_table.size;

    // Search the linked list at the calculated index
    hash_node *current = ht->table[index];

    while (current != NULL) {
        if (strcmp(node_str, current->node_str) == 0) {
            return current->node_index; // Found
        }
        current = current->next;
    }

    return NOT_FOUND; // Not found
}

// This function prints the hash table elements
void print_hash_table(hash_table *ht) {

    printf("Hash table size is %d\n", ht->size);

        for (int i = 0; i < node_hash_table.size; i++) {

            hash_node* current = ht->table[i];

            while (current != NULL) {
                printf("Hash index %d: %s with node index %d\n", i, current->node_str, current->node_index);
                current = current->next;
            }
    }
}

// This function frees the nodes in the hash table
void free_hash_table(hash_table *ht) {

    for (int i = 0; i < node_hash_table.size; i++) {

        hash_node *current = ht->table[i];
        while (current != NULL) {

            hash_node *temp = current;
            current = current->next;
            free(temp->node_str);
            free(temp);
        }
        ht->table[i] = NULL; 
    }

    // Free the table itself
    free(ht->table);
}


// This function adds a node to the node array
void add_node_array(char* node_name)
{
    if(strcmp(node_name,"0")==0){
        return;
    }

    // if it is the first time of adding a node, allocate the array, and add the ground to position 0
    if (node_array == NULL)
    {
        node_array = (char**) malloc(sizeof(char*)*2);
        node_array[0] = (char*) malloc(strlen(node_name) + 1);
        node_array[1] = (char*) malloc(strlen(node_name) + 1);

        // put in the node_array[0] the string "0"
        strcpy(node_array[0], "0");
        strcpy(node_array[1], node_name);
        node_array_index_counter = 2;
        return;
    }
    else
    {
        // realloc the array and add the new node
        node_array = (char**) realloc(node_array, sizeof(char*) * (node_array_index_counter + 1));
        node_array[node_array_index_counter] = (char*) malloc(strlen(node_name) + 1);
        strcpy(node_array[node_array_index_counter], node_name);
        node_array_index_counter++;
    }
}

// This function prints the node array
void print_node_array()
{
    for (int i = 0; i < node_array_index_counter; i++)
    {
        printf("\nThe name of the node with index %d is %s\n", i, node_array[i]);
    }
}

// This function is an array of all V or L elements
void add_m2_array(char* comp_name)
{

    // Allocate the array the first time
    if(m2_array == NULL)
    {
        m2_array = (char**) malloc(sizeof(char*)*(m2_i + 1));
        m2_array[m2_i] = (char*) malloc(strlen(comp_name) + 1);
        strcpy(m2_array[m2_i], comp_name);
        return;
    }
    else
    {
        m2_array = (char**) realloc(m2_array, sizeof(char*) * (m2_i + 1));
        m2_array[m2_i] = (char*) malloc(strlen(comp_name) + 1);
        strcpy(m2_array[m2_i], comp_name);
    }
}

// This function prints the m2 array
void print_m2_array()
{
    for (int i = 0; i < m2_i; i++)
    {
        printf("\nThe name of the component with index %d is %s\n", i, m2_array[i]);
    }
}

// free the node array
void free_node_array()
{
    for (int i = 0; i < node_array_index_counter; i++)
    {
        free(node_array[i]);
    }
    free(node_array);
}

// free the m2 array
void free_m2_array()
{
    for (int i = 0; i < m2; i++)
    {
        free(m2_array[i]);
    }
    free(m2_array);
}

// Function to print all numbers of a transient_spec
void print_spec_numbers(transient_spec_type spec_type, transient_spec *spec) {
    switch (spec_type) {
        case EXP_SPEC:
            printf("EXP_SPEC: %f %f %f %f %f %f\n",
                   spec->exp.i1, spec->exp.i2, spec->exp.td1, spec->exp.tc1, spec->exp.td2, spec->exp.tc2);
            break;
        case SIN_SPEC:
            printf("SIN_SPEC: %f %f %f %f %f %f\n",
                   spec->sin.i1, spec->sin.ia, spec->sin.fr, spec->sin.td, spec->sin.df, spec->sin.ph);
            break;
        case PULSE_SPEC:
            printf("PULSE_SPEC: %f %f %f %f %f %f %f\n",
                   spec->pulse.i1, spec->pulse.i2, spec->pulse.td, spec->pulse.tr, spec->pulse.tf, spec->pulse.pw, spec->pulse.per);
            break;
        case PWL_SPEC:
            printf("PWL_SPEC: %d Pairs\n", spec->pwl.pairs);
            for (int i = 0; i < spec->pwl.pairs; ++i) {
                printf("%f %f\n", spec->pwl.t[i], spec->pwl.i[i]);
            }
            break;
        case AC_SPEC:
            printf("AC_SPEC: %f %f\n", spec->ac.mag, spec->ac.phase);
            break;
        case NO_SPEC:
            printf("NO_SPEC\n");
            break;
        default:
            printf("Unknown spec type\n");
    }
}