FC = mpif90 -ffree-line-length-none

.SUFFIXES: .f90

.f90.o:
	$(FC) -cpp -O3 -fopenmp -DWITH_OPENMP -DWITH_REAL_GENERIC_KERNEL -DWITH_COMPLEX_GENERIC_KERNEL -c -c $<

OBJ = elpa2_kernels_complex.o elpa2_kernels_real.o elpa1.o elpa2.o libelpa.o

all: $(OBJ)
	ar -r libelpa.a *.o
	
clean:
	rm *.o *.a *.mod
