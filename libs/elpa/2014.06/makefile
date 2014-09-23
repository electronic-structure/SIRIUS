include ../../../make.inc

.SUFFIXES: .f90

.f90.o:
	$(MPI_FC) $(MPI_FC_OPT) -DWITH_OPENMP -DWITH_REAL_GENERIC_KERNEL -DWITH_COMPLEX_GENERIC_KERNEL -c -c $<

OBJ = elpa2_kernels_complex.o elpa2_kernels_real.o elpa1.o elpa2.o libelpa.o

all: $(OBJ)
	ar -r libelpa.a *.o
	
clean:
	rm *.o *.a *.mod
