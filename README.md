<p align="center">
<img src="doc/images/sirius_logo.png" width="500">
</p>

[![GitHub Releases](https://img.shields.io/github/release/electronic-structure/sirius.svg)](https://github.com/electronic-structure/SIRIUS/releases)
[![Documentation](https://img.shields.io/badge/docs-doxygen-blue.svg)](https://electronic-structure.github.io/SIRIUS-doc)
[![Licence](https://img.shields.io/badge/license-BSD-blue.svg)](https://raw.githubusercontent.com/electronic-structure/SIRIUS/master/LICENSE)

## Table of contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Examples](#examples)

## Introduction
SIRIUS is a domain specific library for electronic structure calculations. It implements plane wave pseudopotential (PW-PP) and full potential linearized augmented plane wave (FP-LAPW) methods and designed to work with popular community codes such as Exciting, Elk and Quantum ESPRESSO. SIRIUS is written in C++11 with MPI, OpenMP and CUDA programming models.

## Installation
SIRIUS depends on the following libraries: MPI, BLAS, LAPACK, GSL, LibXC, HDF5, spglib, FFTW and optionally on ScaLAPACK, ELPA, MAGMA and CUDA.
We use CMake as a building tool.

### Manual installation
The minimal dependencies dependencies (GSL, LibXC, HDF5, spglib, FFTW) can be downloaded and configured automatically by the helper Python script ``prerequisite.py``, other libraries must be provided by a system or a developer. To compile and install SIRIUS (assuming that all the libraries are installed in the standard paths) run a cmake command from an empty directory followed by a make command:
```console
$ mkdir _build
$ cd _build
$ CXX=mpic++ CC=mpicc FC=mpif90 cmake ../ -DCMAKE_INSTALL_PREFIX=$HOME/local
$ make
$ make install
```
This will compile SIRIUS in a most simple way: CPU-only mode without parallel linear algebra routines.

In order to download and build the compulsory dependencies (xc, gsl, spg, fftw)
the script `prerequisite.py` can be used:

```console
$ mkdir -p libs
$ python prerequisite.py ${PWD}/libs xc spg gsl fftw hdf5
```

Unless the dependencies are installed system wide, set the following
environment variables to the installation path of FFTW, SPGLIB, and LibXC
respectively:
- `FFTWROOT`
- `LIBSPGROOT`
- `LIBXCROOT`
- `HDF5_ROOT`
- `GSL_ROOT_DIR`
- `MAGMAROOT` (optional)
- `MKLROOT` (optional)
- `ELPAROOT` (optional)


### Installation on Piz Daint
We provide an EasyBuild script on Piz Daint. See also the [CSCS EasyBuild Documentation](https://user.cscs.ch/computing/compilation/easybuild/).

```console
# obtain the official CSCS easybuild custom repository
git clone https://github.com/eth-cscs/production
export EB_CUSTOM_REPOSITORY=${HOME}/production/easybuild
module load daint-gpu
module load Easybuild-custom/cscs
# install easybuild package
eb SIRIUS-6.2.0-CrayIntel-18.08-cuda-9.1.eb -r
```

After the installation has completed, the module can be loaded using:
```console
module load SIRIUS/6.2.0-CrayIntel-18.08-cuda-9.1
```
For the SIRIUS enabled version of QE use `eb QuantumESPRESSO-6.4-rc1-sirius-CrayIntel-18.08-cuda-9.1.eb -r`.


### Installation via the Spack package manager
[Spack](https://spack.io) is a package manager for supercomputers, Linux and macOS. It makes installing scientifc software easy.

Install spack (if it is not already on your system):
```console
git clone https://github.com/spack/spack.git
. spack/share/spack/setup-env.sh
```

Clone the SIRIUS spack package repository:
```console
git clone  https://github.com/simonpintarelli/sirius-spack
spack repo add sirius-spack
```

Install SIRIUS (with CUDA support):
```console
spack install sirius +cuda
```
(see `spack info sirius` for all build options).

Load SIRIUS:
```console
spack load sirius +cuda
```

Consult the [Spack documentation](https://spack.readthedocs.io/en/latest/) for more information on how to use Spack.



### Archlinux
Archlinux users can find SIRIUS in the [AUR](https://aur.archlinux.org/packages/sirius-git/).

## Examples
