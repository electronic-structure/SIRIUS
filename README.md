SIRIUS
======
SIRIUS is a domain specific library for electronic structure calculations. It is designed to work with codes such as Exciting, Elk and Quantum ESPRESSO. SIRIUS is written in C++11 with MPI, OpenMP and CUDA programming models. Up-to-date source code documantation is available [here](https://electronic-structure.github.io/SIRIUS-doc/).

## How to compile
There are few prerequisites that you need to know in order to compile and install SIRIUS library. SIRIUS depends on the following libraries: MPI, Blas, Lapack, GSL, LibXC, HDF5, spglib, FFTW and optionally on ScaLAPACK, ELPA, MAGMA, CUDA. Some of the libraries (GSL, LibXC, HDF5, spglib, FFTW) can be downloaded and configured automatically by the install script, other libraries must be provided by the developer. To configure the library you need to provide a JSON file where you specify your compiliers, compiler flags, include paths, and system libraries. Configure script relies on this information!

Configuration JSON file has the following enteries:
  * "MPI_CXX": name of the MPI C++ wrapper (like mpicxx, CC, mpiicc)
  * "MPI_CXX_OPT": C++ compiler options
  * "MPI_FC": name of the MPI Fortran wrapper (like mpif90, ftn mpiifort)  

```JSON

{
    "MPI_FC"      : "ftn",
    "MPI_FC_OPT"  : "-O3 -fopenmp -cpp",

    "comment"     : "plain C compler",
    "CC"          : "cc",

    "comment"     : "plain C++ compiler",
    "CXX"         : "CC",

    "comment"     : "plain Fortran compiler",
    "FC"          : "ftn",

    "comment"     : "Fortran preprocessor",
    "FCCPP"       : "cpp",

    "comment"     : "location of CUDA toolkit",
    "CUDA_ROOT"   : "$(CUDATOOLKIT_HOME)",

    "comment"     : "CUDA compiler and options",
    "NVCC"        : "nvcc",
    "NVCC_OPT"    : "-arch=sm_60 -m64 -DNDEBUG",

    "comment"     : "location of MAGMA library",
    "MAGMA_ROOT"  : "$(HOME)/src/daint/magma-2.2.0",

    "SYSTEM_LIBS" : "$(MKLROOT)/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group  $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_gnu_thread.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group -lpthread -lstdc++ -ldl",

    "install"     : ["spg", "gsl", "xc"]
}
```

