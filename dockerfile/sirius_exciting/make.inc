# NOTE: If compiling with GCC 10 append F90_OPTS and F90_DEBUGOPTS with: 
#  -fallow-argument-mismatch
# See here for more details: https://gcc.gnu.org/gcc-10/porting_to.html

# Compiler, flags and preprocessing variables
F90 = gfortran
F90_OPTS = -O2 -march=x86-64 -ffree-line-length-0
CPP_ON_OPTS = -cpp -DXS -DISO $(SIRIUS_CPP_FLAGS)
F77 = $(F90)
F77_OPTS = -O3

# Required for compilation of LibXC 
FCCPP = cpp

#F90_DEBUGOPTS = -g -O0 -DUSE_ASSERT -fbounds-check -fbacktrace -Wall -Wextra -ffree-line-length-0 -fcheck=all -finit-integer=2147483647 -finit-real=snan \
#-frecord-gcc-switches -finit-character=42 -finit-logical=true -ffpe-trap=invalid,zero,overflow -fdump-core -fstack-protector-all -pipe
# Unusable debug flags: -pedantic 
#F77_DEBUGOPTS = $(F90_DEBUGOPTS)

# If compiling on mac, one may need to specify the location of the openMP header file.
# For example, if using Macports it may look like:
# INCL = -I/opt/local/lib/gcc10/gcc/x86_64-apple-darwin19/10.2.0/finclude/
#INCL =

# Libaries
LIB_ARP = 
# Use native blas/lapack by default
export USE_SYS_LAPACK=true
LIB_LPK = $(LAPACK_LIB)
LIB_FFT = fftlib.a
LIB_BZINT = libbzint.a
LIBS = $(LIB_ARP) $(LIB_LPK) $(LIB_FFT) $(LIB_BZINT) $(SIRIUS_LIB)

# SMP and MPI compilers, flags and preprocessing variables
MPIF90 = mpif90
MPIF90_OPTS = -DMPI
MPI_LIBS =

# To use Scalapack, include the preprocessing variable, provide the library path and library name 
#MPIF90_OPTS = -DMPI -DSCAL
#MPI_LIBS = -L/opt/local/lib/ -lscalapack

SMPF90_OPTS = -fopenmp -DUSEOMP
SMPF77_OPTS = $(SMPF90_OPTS)
SMP_LIBS = 

BUILDMPI = true
BUILDSMP = true
BUILDMPISMP = true

