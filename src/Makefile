include ../make.inc

.SUFFIXES: .cpp .cu

SRC =  LebedevLaikov.cpp linalg.cpp utils.cpp error_handling.cpp timer.cpp platform.cpp radial_grid.cpp \
       radial_solver.cpp sht.cpp atom_type.cpp atom_symmetry_class.cpp atom.cpp unit_cell.cpp \
       reciprocal_lattice.cpp step_function.cpp global.cpp k_point.cpp band.cpp potential.cpp \
       k_set.cpp density.cpp
       
      
OBJ = $(SRC:.cpp=.o) 
      
ifneq (,$(findstring D_GPU_,$(CXX_OPT)))
	OBJ := $(OBJ) cuda_interface.o
endif

ifneq (,$(findstring D_MAGMA_,$(CXX_OPT)))
	OBJ := $(OBJ) magma_interface.o
endif
	
.cpp.o:
	$(CXX) $(CXX_OPT) -c -o $(*D)/$(*F).o $<

.cu.o:
	$(NVCC) -DADD_ $(NVCC_OPT) -c $<

all: log $(OBJ) sirius.o
	ar -r libsirius.a $(OBJ)
	ar -r libsirius_f.a $(OBJ) sirius.o

# TODO: platform-independent python script
log:
	@echo "#ifndef __VERSION_H__" > version.h
	@echo "#define __VERSION_H__" >> version.h
	@echo "const char* const git_hash = \"$(shell git rev-parse HEAD)\";" >> version.h 
	@echo "const char* const build_date = \"$(shell date)\";" >> version.h 
	@echo "#endif" >> version.h

clean:
	rm -f $(OBJ) *.o *.a

depend: $(SRC) sirius.cpp
	rm -f ./.depend
	$(CXX) $(CXX_OPT) -MM $^ >> ./.depend

showlibs:
	@echo
	@echo List of libraries for linking with the Fortran code:
	@echo
	@echo $(shell pwd)/libsirius_f.a $(LIBS) 

ifneq ($(wildcard .depend),)
include .depend
endif
