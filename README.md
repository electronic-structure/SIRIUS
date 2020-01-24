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
* [Examples](#examples)

## Introduction
SIRIUS is a domain specific library for electronic structure calculations. It implements pseudopotential plane wave (PP-PW)
and full potential linearized augmented plane wave (FP-LAPW) methods and is designed for GPU acceleration of popular community
codes such as Exciting, Elk and Quantum ESPRESSO. SIRIUS is written in C++11 with MPI, OpenMP and CUDA/ROCm programming models.
SIRIUS is organised as a collection of classes that abstract away the different building blocks of DFT self-consistency cycle.

## Installation
SIRIUS has a hard dependency on the following libraries: MPI, BLAS, LAPACK, [GSL](https://www.gnu.org/software/gsl/),
[LibXC](https://www.tddft.org/programs/libxc/), [HDF5](https://www.hdfgroup.org/solutions/hdf5/),
[spglib](https://atztogo.github.io/spglib/) and [SpFFT](https://github.com/eth-cscs/SpFFT). They
must be available on your platfrom. Optionally, there is a dependency on ScaLAPACK, [ELPA](https://elpa.mpcdf.mpg.de/software),
[MAGMA](https://icl.cs.utk.edu/magma/) and CUDA/ROCm. We use CMake as a building tool. If the libraries are installed
in a standard location, cmake can find them automatically.  Otherwise you need to provide a specific path of each
library to cmake. We use Docker to create a reproducible work environment for the examples below.

### Minimal installation
Suppose we have a minimal Linux installation described the following Dockerfile:
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
We can then execute the following set of commands inside the docker container:
```console
git clone --recursive https://github.com/electronic-structure/SIRIUS.git
cd SIRIUS
CC=mpicc CXX=mpic++ FC=mpif90 FCCPP=cpp FFTW_ROOT=$HOME/local python3 prerequisite.py $HOME/local fftw spfft gsl hdf5 xc spg
mkdir build
cd build
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
```console
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

SIRIUS can be build inside this docker container using the following command:
```console
git clone --recursive https://github.com/electronic-structure/SIRIUS.git
mkdir SIRIUS/build
cd SIRIUS/build
cmake .. -DUSE_SCALAPACK=1 -DBUILD_TESTS=1 -DCREATE_PYTHON_MODULE=1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/local
make -j install

```

You can also use Spack to install SIRIUS. For example:
```console
# install SIRIUS with CUDA support
spack install sirius +cuda
```
(see `spack info sirius` for all build options).

Load SIRIUS:
```console
spack load -r sirius +cuda
```

Please refere to [Spack documentation](https://spack.readthedocs.io/en/latest/) for more information on how to use Spack.



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
Please refer to [SIRIUS wiki page](https://github.com/electronic-structure/SIRIUS/wiki/Build-on-Piz-Daint) for
specific details.



For the SIRIUS enabled version of QE use `eb QuantumESPRESSO-6.4-rc3-sirius-CrayIntel-19.10-cuda-10.1.eb -r`.


## Examples
