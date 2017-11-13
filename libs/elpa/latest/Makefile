include ../../../make.inc

external_elpa_lib = false
ifeq ($(external_elpa_lib), true)
  MPI_FC_OPT := $(MPI_FC_OPT) -D__EXTERNAL_ELPA_LIB -I/Users/antonk/local/include/elpa_openmp-2017.05.002/modules
endif

.SUFFIXES: .F90 .f90

.F90.o:
	$(MPI_FC) $(MPI_FC_OPT) -c $<

.f90.o:
	$(MPI_FC) $(MPI_FC_OPT) -c $<

ifeq ($(external_elpa_lib), true)
OBJ = elpa_api.o
else
OBJ = mod_precision.o elpa_utilities.o elpa2_utilities.o mod_mpi.o elpa1_compute_private.o elpa1_auxiliary.o elpa1.o qr_utils.o \
      elpa_pdlarfb.o elpa_pdgeqrf.o aligned_mem.o \
      elpa2_kernels_complex.o elpa2_kernels_complex_simple.o elpa2_kernels_real.o elpa2_kernels_real_simple.o \
      elpa_qrkernels.o mod_pack_unpack_real.o mod_pack_unpack_complex.o mod_single_hh_trafo_real.o \
      mod_compute_hh_trafo_real.o mod_compute_hh_trafo_complex.o \
      elpa2_compute.o elpa2.o elpa_api.o
endif

all: $(OBJ)
	ar -r libelpa.a *.o
	
clean:
	rm *.o *.a *.mod
