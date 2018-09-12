SIRIUS
======
SIRIUS is a domain specific library for electronic structure calculations. It is designed to work with codes such as Exciting, Elk and Quantum ESPRESSO. SIRIUS is written in C++11 with MPI, OpenMP and CUDA programming models. Up-to-date source code documantation is available [here](https://electronic-structure.github.io/SIRIUS-doc/).

## How to compile
There are few prerequisites that you need to know in order to compile and install SIRIUS library. SIRIUS depends on the following libraries: MPI, Blas, Lapack, GSL, LibXC, HDF5, spglib, FFTW and optionally on ScaLAPACK, ELPA, MAGMA, CUDA. Some of the libraries (GSL, LibXC, HDF5, spglib, FFTW) can be downloaded and configured automatically by the install script, other libraries must be provided by the developer. To configure the library you need to provide a JSON file where you specify your compiliers, compiler flags, include paths, and system libraries. Configure script relies on this information!

Configuration JSON file has the following enteries:
  * "MPI_CXX": name of the MPI C++ wrapper (like mpicxx, CC, mpiicc)
  * "MPI_CXX_OPT": C++ compiler options
  * "MPI_FC": name of the MPI Fortran wrapper (like mpif90, ftn, mpiifort)
  * "MPI_FC_OPT": Fortran compiler options
  * "CC": plain C compiler (for example gcc, clang, icc)
  * "CXX": plain C++ compiler (for example g++, clang++, icpc)
  * "FC": plain Forrtan compiler (for example gfortran, ifort)
  * "FCCPP": Fortran preprocessor (usually cpp)
  * "SYSTEM_LIBS": list of the libraries necessary for linking
  * "install": list of the packages to download and configure automatically (can be any of "fftw", "xc", "spg", "hdf5", "gsl")
  
Optionally, the following enteries can be provided:
  * "CUDA_ROOT": path to a CUDA toolkit installation
  * "CUDA_CC": name of the CUDA compiler (if none is provided, nvcc will be used by default)
  * "CUDA_OPT": CUDA compiler options
  * "MAGMA_ROOT": path to a compiled MAGMA library
  * "ELPA_ROOT": path to a compiled ELPA library
