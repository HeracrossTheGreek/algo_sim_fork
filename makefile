# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -g -std=c11

# Include directories
INCLUDES = -I/home/makaragiannis/gsl/include -I/home/makaragiannis/Documents/CXSparse-master/Include

# Libraries
LIBS = -L/home/makaragiannis/gsl/lib -lgsl -lgslcblas -lm -L/home/makaragiannis/Documents/CXSparse-master/Lib -lcxsparse

# Source files
SRC = main.c parse.c structs.c mna.c direct_sol.c iter_sol.c gsl.c csparse.c sparse_sol.c transient.c ac.c

# Object files directory
OBJ_DIR = obj

# Object files (derived from source files)
OBJ = $(addprefix $(OBJ_DIR)/,$(SRC:.c=.o))

# Executable name
EXEC = spice_sim

# Main target
all: $(EXEC)

# Create the obj directory
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Compile source files into object files
$(OBJ_DIR)/%.o: %.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

# Link object files to create the executable
$(EXEC): $(OBJ)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(EXEC) $(OBJ) $(LIBS)

# Clean up object files and executables
clean:
	rm -rf $(OBJ_DIR) $(EXEC)
