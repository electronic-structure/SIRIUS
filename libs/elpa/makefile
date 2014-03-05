FC = mpif90

all:
	$(FC) -cpp -O3 -ffree-line-length-none -fopenmp -DWITH_OPENMP -c *90
	ar -r libelpa.a *.o
	
clean:
	rm *.o *.a *.mod
