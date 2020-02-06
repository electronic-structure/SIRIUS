<p align="center">
<img src="doc/images/sirius_logo.png" width="500">
</p>

[![GitHub Releases](https://img.shields.io/github/release/electronic-structure/sirius.svg)](https://github.com/electronic-structure/SIRIUS/releases)
[![Documentation](https://img.shields.io/badge/docs-doxygen-blue.svg)](https://electronic-structure.github.io/SIRIUS-doc)
[![Licence](https://img.shields.io/badge/license-BSD-blue.svg)](https://raw.githubusercontent.com/electronic-structure/SIRIUS/master/LICENSE)

## Table of contents
* [Introduction](#introduction)
* [Installation](#installation)
  * [Minimal installation](#minimal-installation)
  * [Installation using Spack](#installation-using-spack)
  * [Adding GPU support](#adding-gpu-support)
  * [Parallel eigensolvers](#parallel-eigensolvers)
  * [Python module](#python-module)
  * [Additional options](#additional-options)
  * [Archlinux](#archlinux)
  * [Installation on Piz Daint](#installation-on-piz-daint)
* [Accelerating DFT codes](#accelerating-dft-codes)
  * [Quantum ESPRESSO](#quantum-espresso)
* [Examples](#examples)

## Introduction
SIRIUS is a domain specific library for electronic structure calculations. It implements pseudopotential plane wave (PP-PW)
and full potential linearized augmented plane wave (FP-LAPW) methods and is designed for GPU acceleration of popular community
codes such as Exciting, Elk and Quantum ESPRESSO. SIRIUS is written in C++11 with MPI, OpenMP and CUDA/ROCm programming models.
SIRIUS is organised as a collection of classes that abstract away the different building blocks of DFT self-consistency cycle.

The following functionality is currently implemented in SIRIUS:
 * (PP-PW) Norm-conserving, ultrasoft and PAW pseudopotentials
 * (PP-PW) Spin-orbit coupling
 * (PP-PW) Stress tensor
 * (PP-PW, FP-LAPW) Atomic forces
 * (PP-PW, FP-LAPW) Collinear and non-collinear magnetism
 * (FP-LAPW) APW and LAPW basis sets with arbitray number of local orbitals
 * (FP-LAPW) ZORA and IORA approximations for valence states; full relativistic Dirac equation for core states
 * Python frontend
 * Symmetrization of lattice-periodic functions and on-site matrices
 * Generation of irreducible k-meshes


## Installation
SIRIUS has a hard dependency on the following tools and libraries:
 * CMake >= 3.14
 * C++ compiler with C++11 support
 * MPI
 * BLAS/LAPACK
 * [GSL](https://www.gnu.org/software/gsl/) - GNU scientifc library
 * [LibXC](https://www.tddft.org/programs/libxc/) - library of exchange-correlation potentials
 * [HDF5](https://www.hdfgroup.org/solutions/hdf5/)
 * [spglib](https://atztogo.github.io/spglib/) - library for finding and handling crystal symmetries
 * [SpFFT](https://github.com/eth-cscs/SpFFT) - domain-specific FFT library

They must be available on your platfrom. Optionally, there is a dependency on:
 * ScaLAPACK
 * [ELPA](https://elpa.mpcdf.mpg.de/software)
 * [MAGMA](https://icl.cs.utk.edu/magma/)
 * CUDA/ROCm

We use CMake as a building tool. If the libraries are installed in a standard location, cmake can find them
automatically.  Otherwise you need to provide a specific path of each library to cmake. We use Docker to create a
reproducible work environment for the examples below.

### Minimal installation
Suppose we have a minimal Linux installation described by the following Dockerfile:
```dockerfile
FROM ubuntu:bionic

ENV DEBIAN_FRONTEND noninteractive

ENV FORCE_UNSAFE_CONFIGURE 1

RUN apt-get update

RUN apt-get install -y apt-utils

# install basic tools
RUN apt-get install -y gcc g++ gfortran mpich git make \
    vim wget pkg-config python3 curl liblapack-dev \
    apt-transport-https ca-certificates gnupg software-properties-common

# install latest CMake (needed by SIRIUS and SpFFT)
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update
RUN apt-get install -y cmake
RUN apt-get install -y kitware-archive-keyring
RUN apt-key --keyring /etc/apt/trusted.gpg del C1F34CDD40CD72DA

WORKDIR /root
ENTRYPOINT ["bash", "-l"]
```
We can then build SIRIUS with the following set of commands:
```bash
git clone --recursive https://github.com/electronic-structure/SIRIUS.git
cd SIRIUS
# build dependencies (spfft, gsl, hdf5, xc, spg) and install them to $HOME/local
CC=mpicc CXX=mpic++ FC=mpif90 FCCPP=cpp FFTW_ROOT=$HOME/local python3 prerequisite.py $HOME/local fftw spfft gsl hdf5 xc spg
mkdir build
cd build
# configure SIRIUS; search for missing libraries in $HOME/local
CXX=mpicxx CC=mpicc FC=mpif90 GSL_ROOT_DIR=$HOME/local LIBXCROOT=$HOME/local LIBSPGROOT=$HOME/local HDF5_ROOT=$HOME/local cmake ../ -DSpFFT_DIR=$HOME/local/lib/cmake/SpFFT -DCMAKE_INSTALL_PREFIX=$HOME/local
make -j install
```
This will clone SIRIUS repository, install the compulsory dependencies (LibXC, GSL, spglib, SpFFT, HDF5) with the
provided Python script ``prerequisite.py`` and then configure, make and install SIRIUS libray itself in a most simple
configuration with CPU-only mode without parallel linear algebra routines.

Unless the dependencies are installed system wide, set the following environment variables to the installation path of
the corresponding libraries:
- `LIBSPGROOT`
- `LIBXCROOT`
- `HDF5_ROOT`
- `GSL_ROOT_DIR`
- `MAGMAROOT` (optional)
- `MKLROOT` (optional)
- `ELPAROOT` (optional)

### Installation using Spack
[Spack](https://spack.io) is a package manager for supercomputers, Linux and macOS. It is a great tool to manage
complex scientifc software installations. Install Spack (if it is not already on your system):
```bash
git clone https://github.com/spack/spack.git
. spack/share/spack/setup-env.sh
```

In the following Dockerfile example most of the software is installed using Spack:
```dockerfile
FROM ubuntu:bionic

ENV DEBIAN_FRONTEND noninteractive

ENV FORCE_UNSAFE_CONFIGURE 1

RUN apt-get update

RUN apt-get install -y apt-utils

# install basic tools
RUN apt-get install -y gcc g++ gfortran git make cmake unzip \
  vim wget pkg-config python3-pip curl environment-modules tcl

# get latest version of spack
RUN cd && git clone https://github.com/spack/spack.git

ENV SPACK_ROOT /root/spack

# add environment variables
RUN echo "source /root/spack/share/spack/setup-env.sh" >> /etc/profile.d/spack.sh

# build GCC
RUN /bin/bash -l -c "spack install gcc@9.2.0"

# add GCC to environment
RUN echo "spack load --dependencies gcc@9.2.0" >> /etc/profile.d/spack.sh

# update list of spack compilers
RUN /bin/bash -l -c "spack compiler find"

# build CMake
RUN /bin/bash -l -c "spack install --no-checksum cmake@3.16.2 %gcc@9.2.0"

# build other packages
RUN /bin/bash -l -c "spack install --no-checksum py-mpi4py %gcc@9.2.0"
RUN /bin/bash -l -c "spack install --no-checksum netlib-scalapack ^openblas threads=openmp ^cmake@3.16.2 %gcc@9.2.0"
RUN /bin/bash -l -c "spack install --no-checksum hdf5+hl %gcc@9.2.0"
RUN /bin/bash -l -c "spack install --no-checksum libxc %gcc@9.2.0"
RUN /bin/bash -l -c "spack install --no-checksum spglib %gcc@9.2.0"
RUN /bin/bash -l -c "spack install --no-checksum gsl %gcc@9.2.0"
RUN /bin/bash -l -c "spack install --no-checksum spfft %gcc@9.2.0"

RUN echo "spack load --dependencies cmake@3.16.2 %gcc@9.2.0" >> /etc/profile.d/spack.sh
RUN echo "spack load --dependencies netlib-scalapack %gcc@9.2.0" >> /etc/profile.d/spack.sh
RUN echo "spack load --dependencies libxc %gcc@9.2.0" >> /etc/profile.d/spack.sh
RUN echo "spack load --dependencies spglib %gcc@9.2.0" >> /etc/profile.d/spack.sh
RUN echo "spack load --dependencies py-mpi4py %gcc@9.2.0" >> /etc/profile.d/spack.sh
RUN echo "spack load --dependencies hdf5 %gcc@9.2.0" >> /etc/profile.d/spack.sh
RUN echo "spack load --dependencies gsl %gcc@9.2.0" >> /etc/profile.d/spack.sh
RUN echo "spack load --dependencies spfft %gcc@9.2.0" >> /etc/profile.d/spack.sh

WORKDIR /root

ENTRYPOINT ["bash", "-l"]
```

SIRIUS can be built inside this docker container using the following command:
```bash
git clone --recursive https://github.com/electronic-structure/SIRIUS.git
mkdir SIRIUS/build
cd SIRIUS/build
cmake .. -DUSE_SCALAPACK=1 -DBUILD_TESTS=1 -DCREATE_PYTHON_MODULE=1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/local
make -j install

```

You can also use Spack to install SIRIUS. For example:
```bash
# install SIRIUS with CUDA support
spack install sirius +cuda
```
(see `spack info sirius` for all build options).

To load SIRIUS you need to run:
```bash
spack load -r sirius +cuda
```

Please refer to [Spack documentation](https://spack.readthedocs.io/en/latest/) for more information on how to use Spack.


### Adding GPU support
To enable CUDA you need to pass the following options to cmake: `-DUSE_CUDA=On -DGPU_MODEL='P100'`. The currently
supported GPU models are `P100`, `V100` and `G10x0` but other architectures can be added easily. If CUDA is installed in a
non-standard directory, you have to pass additional parameter to cmake `-DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda`.

To enable MAGMA (GPU implementation of Lapack) you need to pass the following option to cmake: `-DUSE_MAGMA=On`. If MAGMA
was installed in a non-standard directory you need to export additional environment variable `MAGMAROOT=/path/to/magma`.

### Parallel eigensolvers
To compile with ScaLAPACK use the following option: `-DUSE_SCALAPACK=On`. Additional environment variable `SCALAPACKROOT`
might be required to specify the location of ScaLAPACK library. To compile with ELPA use `-DUSE_SCALAPACK=On -DUSE_ELPA=On` options.
In this case additional environment variable `ELPAROOT` might be required. In the current implentation we need
ScaLAPACK functionality to transform generalized eigenvalue problem to a standard form bofore using ELPA.

### Python module
To create Python module you need to specify `-DCREATE_PYTHON_MODULE=On`. SIRIUS Python module depends on `mpi4py` and
`pybind11` packages. They must be installed on your platform.

### Additional options
To link against MKL you need to specify `-DUSE_MKL=On` parameter. For Cray libsci use `-DUSE_CRAY_LIBSCI=On`. To build
tests you need to specify `-DBUILD_TESTS=On`.


### Archlinux
Archlinux users can find SIRIUS in the [AUR](https://aur.archlinux.org/packages/sirius-git/).

### Installation on Piz Daint
Please refer to [SIRIUS wiki page](https://github.com/electronic-structure/SIRIUS/wiki/Build-on-Piz-Daint) and 
[CSCS User portal](https://user.cscs.ch/computing/applications/sirius/) for detailed instructions.

## Accelerating DFT codes

### Quantum ESPRESSO
[Quantum ESPRESSO](https://www.quantum-espresso.org/) is a popular open source suite of computer codes for
electronic-structure calculations and materials modeling at the nanoscale. It is based on DFT, plane waves, and
pseudopotentials. We maintain the GPU-accelerated version of 
[Quantum ESPRESSO with SIRIUS bindings](https://github.com/electronic-structure/q-e-sirius).
This version is frequently synchronised with the
`develop` branch of the official [QE repository](https://gitlab.com/QEF/q-e). A typical example of using SIRIUS
inside QE looks like this:
```Fortran
IF (use_sirius.AND.use_sirius_vloc) THEN
  ALLOCATE(tmp(ngm))
  CALL sirius_get_pw_coeffs_real(sctx, atom_type(nt)%label, string("vloc"), tmp(1), ngm, mill(1, 1), intra_bgrp_comm)
  DO i = 1, ngm
    vloc(igtongl(i), nt) = tmp(i) * 2 ! convert to Ry
  ENDDO
  DEALLOCATE(tmp)
ELSE
CALL vloc_of_g( rgrid(nt)%mesh, msh(nt), rgrid(nt)%rab, rgrid(nt)%r, &
                upf(nt)%vloc(1), upf(nt)%zp, tpiba2, ngl, gl, omega, &
                vloc(1,nt) )
ENDIF ! sirius

```
To compile QE+SIRIUS you need to go through this basic steps:
 * compile and install SIRIUS
 * configure QE+SIRIUS
 * `make pw`

The behaviour of QE configuration script changes from time to time, so you have to figure out how it works on your
system. As a starting point, try this set of commands:
```bash
git clone --recursive -b qe_sirius https://github.com/electronic-structure/q-e-sirius.git
cd ./q-e-sirius
CC=mpicc FC=mpif90 LIBS="-L$/path/to/sirius/lib -Wl,-rpath,/path/to/sirius/lib -lsirius -lpthread -fopenmp" \
  LDFLAGS=$LIBS LD_LIBS=$LIBS F90FLAGS="-I/path/to/sirius/include -I$MKLROOT/include/fftw" \
  ./configure --enable-openmp --enable-parallel --with-scalapack

# sometimes this is also needed if BLAS/LAPACK provider is not recognized properly
sed -i -e "/LAPACK_LIBS    =/d" make.inc
sed -i -e "s/LAPACK_LIBS_SWITCH = internal/LAPACK_LIBS_SWITCH = external/" make.inc

make -j pw
```
This should hopefully produce the `pw.x` binary in `PW/src` folder. If this doesn't work, try to configure QE as you 
usually do and then modify `make.inc` file by hand to add `-I/path/to/sirius/include` directory to the Fortran compiler
options and `-L$/path/to/sirius/lib -Wl,-rpath,/path/to/sirius/lib -lsirius` to the linker flags.

Once `pw.x` binary is created, you can run it with the same parameters and input file as you run the native QE.
By default, SIRIUS library is not used. To enable SIRIUS pass command-line option `-sirius` to `pw.x`.

```bash
# run in default mode
pw.x -i pw.in
# run with SIRIUS enabled
pw.x -i pw.in -sirius
```

SIRIUS library is usgin OpenMP for node-level parallelization. To run QE/SIRIUS efficiently, follow these simple rules:
 * always prefer k-point pool parallelization over band parallelization
 * use as few MPI ranks as possible for band parallelization
 * by default, use one rank per node and many OMP threads; if the calculated system is really small, try to saturate 
   the GPU card using more MPI ranks (e.g.: on a 12-core node, use 2-3-4 ranks with 6-4-3 OMP threads)

#### Example: ground state of Si511Ge
In the following example we compare performace of native and SIRIUS-enabled versions of QE. Native QE was run on the
dual-socket nodes containing two 18-core Intel Haswell CPUs.


<p align="center">
<img src="doc/images/Si511Ge_perf.png">
</p>




## Examples
