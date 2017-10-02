include ../../make.inc

.SUFFIXES: .cpp .cu .f90

SRC = sddk_api.cpp

OBJ = $(SRC:.cpp=.o) sddk.o

ifneq (,$(findstring D__GPU,$(CXX_OPT)))
	OBJ := $(OBJ) \
	GPU/fft_kernels.o \
	GPU/checksum.o \
	GPU/scale_matrix.o
endif

.cpp.o:
	$(CXX) $(CXX_OPT) -c -o $(*D)/$(*F).o $<

.cu.o:
	$(NVCC) $(NVCC_OPT) -c -o $(*D)/$(*F).o $<

.f90.o:
	$(MPI_FC) $(MPI_FC_OPT) -c -o $(*D)/$(*F).o $<

all: $(OBJ)
	ar -r libsddk.a $(OBJ)
	$(MPI_FC) $(MPI_FC_OPT) -c sddk.f90 -o sddk.o
	$(MPI_FC) $(MPI_FC_OPT) test.f90 -o test libsddk.a -lstdc++ $(LIBS)
	
#all:
#	$(CXX) $(CXX_OPT) -c sddk_api.cpp -o sddk_api.o
#	$(MPI_FC) $(MPI_FC_OPT) -c sddk.f90 -o sddk.o
#	$(MPI_FC) $(MPI_FC_OPT) test.f90 -o test sddk_api.o sddk.o -lstdc++ /Users/antonk/src/github/toxa81/SIRIUS/libs/fftw-3.3.5/.libs/libfftw3.a
	
clean:
	rm *.o *.mod GPU/*.o
