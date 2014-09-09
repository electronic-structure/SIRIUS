FC = ftn -xAVX -mavx -vec-report=7 -opt-report=3 -cpp -O3 -openmp

.SUFFIXES: .f90

.f90.o:
	$(FC) -DWITH_OPENMP -DWITH_REAL_GENERIC_KERNEL -DWITH_COMPLEX_GENERIC_KERNEL -c -c $<

OBJ = elpa2_kernels_complex.o elpa2_kernels_real.o elpa1.o elpa2.o libelpa.o

all: $(OBJ)
	ar -r libelpa.a *.o
	
clean:
	rm *.o *.a *.mod
