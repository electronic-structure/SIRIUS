include ../../make.inc

.SUFFIXES: .cpp .cu

NVCC = nvcc

OBJ = linalg_cpu.o jsoncpp.o
      
ifneq (,$(findstring D_GPU_,$(CPP_OPT)))
	OBJ := $(OBJ) gpu_interface.o
endif
	
.cpp.o:
	$(CXX) $(CXX_OPT) -c $<

.cu.o:
	$(NVCC) -DADD_ -I/project/s299/magma/magma-1.1_my-xk6-gcc-4.6.2-libsci/include -c $<

all: $(OBJ)
	ar -r libsirius.a $(OBJ)

clean:
	rm *.o *.a
