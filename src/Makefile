include ../make.inc

.SUFFIXES: .cpp .cu

SRC = ./LebedevLaikov.cpp linalg.cpp utils.cpp error_handling.cpp timer.cpp platform.cpp radial_grid.cpp radial_solver.cpp sht.cpp
      
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
	@echo "const char* git_hash = \"$(shell git rev-parse HEAD)\";" > version.h 
	@echo "const char* build_date = \"$(shell date)\";" >> version.h 

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
