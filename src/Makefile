include ../make.inc

.SUFFIXES: .cpp .cu

NVCC = nvcc

SRC = ./LebedevLaikov.cpp
      
OBJ = $(SRC:.cpp=.o) 
      
ifneq (,$(findstring D_GPU_,$(CPP_OPT)))
	OBJ := $(OBJ) gpu_interface.o
endif
	
.cpp.o:
	$(CXX) $(CXX_OPT) -c -o $(*D)/$(*F).o $<

.cu.o:
	$(NVCC) -DADD_ -I/project/s299/magma/magma-1.1_my-xk6-gcc-4.6.2-libsci/include -c $<

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
	@echo $(LIBS) $(shell pwd)/libsirius.a 

ifneq ($(wildcard .depend),)
include .depend
endif
