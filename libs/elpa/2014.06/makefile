include ../../../make.inc

.SUFFIXES: .f90

.f90.o:
	$(MPI_FC) $(MPI_FC_OPT) -DWITH_OPENMP -DWITH_REAL_GENERIC_KERNEL -DWITH_COMPLEX_GENERIC_KERNEL -c -c $<

OBJ = elpa_utilities.o elpa2_utilities.o elpa1.o qr_utils.o elpa_pdlarfb.o elpa_pdgeqrf.o elpa2_kernels_complex.o \
      elpa2_kernels_real.o elpa2.o libelpa.o elpa_qrkernels.o

all: $(OBJ)
	ar -r libelpa.a *.o
	
clean:
	rm *.o *.a *.mod
