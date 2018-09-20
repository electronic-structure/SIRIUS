SIRIUS
======
SIRIUS is a domain specific library for electronic structure calculations. It is designed to work with codes such as Exciting, Elk and Quantum ESPRESSO. SIRIUS is written in C++11 with MPI, OpenMP and CUDA programming models. Up-to-date source code documantation is available [here](https://electronic-structure.github.io/SIRIUS-doc/).

## How to compile
SIRIUS depends on the following libraries: MPI, BLAS, LAPACK, GSL, LibXC, HDF5, spglib, FFTW and optionally on ScaLAPACK, ELPA, MAGMA and CUDA. Some of the libraries (GSL, LibXC, HDF5, spglib, FFTW) can be downloaded and configured automatically by the helper Python script ``prerequisite.py``, other libraries must be provided by a system or a developer. We use CMake as a building tool. 

Follow this basic steps in order to compile and install SIRIUS.

1. Create a build directory:
```console
$ mkdir _build
```

2. Assuming that all the libraries are installed in the standard paths, run a cmake command from that directory:
```console
$ cd _build
$ CXX=mpic++ CC=mpicc FC=mpif90 cmake /path/to/SIRIUS -DCMAKE_INSTALL_PREFIX=$HOME/local
```

This will configure SIRIUS in a most simple way: CPU-only code without parallel linear algebra routines.

3. Now you can build the whole project:
```console
$ make
```
