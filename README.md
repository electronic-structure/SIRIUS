<p align="center">
<img src="doc/images/sirius_logo.png" width="500">
</p>

[![GitHub Releases](https://img.shields.io/github/release/electronic-structure/sirius.svg)](https://github.com/electronic-structure/SIRIUS/releases)
[![Documentation](https://img.shields.io/badge/docs-doxygen-blue.svg)](https://electronic-structure.github.io/SIRIUS-doc)
[![Licence](https://img.shields.io/badge/license-BSD-blue.svg)](https://github.com/electronic-structure/SIRIUS/LICENSE)


SIRIUS is a domain specific library for electronic structure calculations. It is designed to work with codes such as Exciting, Elk and Quantum ESPRESSO. SIRIUS is written in C++11 with MPI, OpenMP and CUDA programming models.

## How to compile
SIRIUS depends on the following libraries: MPI, BLAS, LAPACK, GSL, LibXC, HDF5, spglib, FFTW and optionally on ScaLAPACK, ELPA, MAGMA and CUDA. Some of the libraries (GSL, LibXC, HDF5, spglib, FFTW) can be downloaded and configured automatically by the helper Python script ``prerequisite.py``, other libraries must be provided by a system or a developer. We use CMake as a building tool. To compile and install SIRIUS (assuming that all the libraries are installed in the standard paths) run a cmake command from an empty directory followed by a make command:

```console
$ mkdir _build
$ cd _build
$ CXX=mpic++ CC=mpicc FC=mpif90 cmake ../ -DCMAKE_INSTALL_PREFIX=$HOME/local
$ make
$ make install
```
This will compile SIRIUS in a most simple way: CPU-only mode without parallel linear algebra routines.
