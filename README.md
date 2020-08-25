<p align="center">
<img src="doc/images/sirius_logo.png" width="500">
</p>

[![GitHub Releases](https://img.shields.io/github/release/electronic-structure/sirius.svg)](https://github.com/electronic-structure/SIRIUS/releases) 
[![Documentation](https://img.shields.io/badge/docs-doxygen-blue.svg)](https://electronic-structure.github.io/SIRIUS-doc)
[![Licence](https://img.shields.io/badge/license-BSD-blue.svg)](https://raw.githubusercontent.com/electronic-structure/SIRIUS/master/LICENSE)
[![Build](https://github.com/electronic-structure/SIRIUS/workflows/Build/badge.svg?branch=master)](https://github.com/electronic-structure/SIRIUS/actions)
[![Verification tests master](https://gitlab.com/cscs-ci/electronic-structure/SIRIUS/badges/master/pipeline.svg?key_text=master)](https://gitlab.com/cscs-ci/electronic-structure/SIRIUS/-/commits/master)
[![Verification tests develop](https://gitlab.com/cscs-ci/electronic-structure/SIRIUS/badges/develop/pipeline.svg?key_text=develop)](https://gitlab.com/cscs-ci/electronic-structure/SIRIUS/-/commits/develop)


## Table of contents
* [Introduction](#introduction)
* [Installation](#installation)
  * [Configuring SIRIUS](#configuring-sirius)
  * [Developing and debugging SIRIUS](#developing-and-debugging-sirius)
  * [Manual installation](#manual-installation)
  * [Archlinux](#archlinux)
  * [Installation on Piz Daint](#installation-on-piz-daint)
* [Accelerating DFT codes](#accelerating-dft-codes)
  * [Quantum ESPRESSO](#quantum-espresso)
  * [CP2K](#cp2k)
* [Contacts](#contacts)
* [Acknowledgements](#acknowledgements)

## Introduction
SIRIUS is a domain specific library for electronic structure calculations. It implements pseudopotential plane wave (PP-PW)
and full potential linearized augmented plane wave (FP-LAPW) methods and is designed for GPU acceleration of popular community
codes such as Exciting, Elk and Quantum ESPRESSO. SIRIUS is written in C++14 with MPI, OpenMP and CUDA/ROCm programming models.
SIRIUS is organised as a collection of classes that abstract away the different building blocks of DFT self-consistency cycle.

The following functionality is currently implemented in SIRIUS:
 * (PP-PW) Norm-conserving, ultrasoft and PAW pseudopotentials
 * (PP-PW) Spin-orbit coupling
 * (PP-PW) Stress tensor
 * (PP-PW, FP-LAPW) Atomic forces
 * (PP-PW, FP-LAPW) Collinear and non-collinear magnetism
 * (FP-LAPW) APW and LAPW basis sets with arbitrary number of local orbitals
 * (FP-LAPW) ZORA and IORA approximations for valence states; full relativistic Dirac equation for core states
 * Python frontend
 * Symmetrization of lattice-periodic functions and on-site matrices
 * Generation of irreducible k-meshes

## Installation
It is recommended to install SIRIUS through [Spack](https://spack.readthedocs.io/en/latest/getting_started.html). To set it up, use

```bash
git clone https://github.com/spack/spack.git
. spack/share/spack/setup-env.sh
spack install sirius
```

### Configuring SIRIUS

SIRIUS has many different configurations to enable specific hardware and library support. Some common configurations include:

```bash
# Use default BLAS, LAPACK, MPI and FFTW3 implementations, without GPU support, using the latest GCC 9.x
$ spack install sirius %gcc@:9

# Explicitly use the latest 3.x release of MPICH for MPI, OpenBLAS for BLAS and LAPACK, FFTW for FFTW3, without GPU support
$ spack install sirius ^mpich@:3 ^fftw ^openblas threads=openmp

# Enable distributed linear algebra, and use Intel MKL for BLAS, ScaLAPACK and FFTW3, without GPU support
$ spack install sirius +scalapack ^intel-mkl

# Build with CUDA support for NVIDIA GPUs
$ spack install sirius +cuda cuda_arch=75

# Build with ROCm support for AMD GPUs
$ spack install sirius +rocm amdgpu_target=gfx906

# Build with MAGMA
$ spack install sirius +cuda +magma

# Build with ELPA
$ spack install sirius +scalapack +elpa
```

Language interop with Fortran and Python can be enabled with `+fortran` and `+python` respectively.

Check out `spack info sirius` for the full list of support variants.

### Developing and debugging SIRIUS

The recommended way to install the latest development version of SIRIUS is through `spack dev-build`.

As an example, the following builds SIRIUS with CUDA support in debug mode:

```bash
$ git clone --recursive -b develop https://github.com/electronic-structure/SIRIUS.git
$ cd SIRIUS
$ spack dev-build sirius@develop build_type=Debug +cuda
```

When you need more control over the build commands, use `spack build-env [spec] -- [command]`:

```bash
$ mkdir SIRIUS/build && cd SIRIUS/build
$ spack install --only=dependencies sirius@develop build_type=Debug +cuda
$ spack build-env sirius@develop build_type=Debug +cuda -- cmake ..
$ spack build-env sirius@develop build_type=Debug +cuda -- make -j$(nproc)
```

### Manual installation

When installing SIRIUS without spack, make sure to install the required dependencies first:

 * CMake ≥ 3.14
 * C++ compiler with C++14 support
 * MPI (OpenMPI or MPICH)
 * BLAS/LAPACK (OpenBLAS or Intel MKL)
 * [GSL](https://www.gnu.org/software/gsl/) - GNU scientific library
 * [LibXC](https://www.tddft.org/programs/libxc/) - library of exchange-correlation potentials
 * [HDF5](https://www.hdfgroup.org/solutions/hdf5/)
 * [spglib](https://atztogo.github.io/spglib/) - library for finding and handling crystal symmetries
 * [SpFFT](https://github.com/eth-cscs/SpFFT) - domain-specific FFT library

and optionally the following libraries:
 * ScaLAPACK (Intel MKL or netlib scalapack)
 * [ELPA](https://elpa.mpcdf.mpg.de/software)
 * [MAGMA](https://icl.cs.utk.edu/magma/)
 * CUDA/ROCm

When libraries are installed in non-standard locations, set `CMAKE_PREFIX_PATH` to a list of installation paths.

```bash
git clone --recursive https://github.com/electronic-structure/SIRIUS.git
mkdir SIRIUS/build
cd SIRIUS/build
export CXX=mpicxx CC=mpicc FC=mpif90
export CMAKE_PREFIX_PATH="path/to/BLAS;path/to/GSL;path/to/LibXC;path/to/HDF5;..."
cmake -DCMAKE_INSTALL_PREFIX=$PWD/sirius
make -j install
```

#### Adding GPU support
To enable CUDA you need to pass the following options to CMake: `-DUSE_CUDA=On -DGPU_MODEL='P100'`. The currently
supported GPU models are `P100`, `V100` and `G10x0` but other architectures can be added easily. If CUDA is installed in a
non-standard directory, you have to pass additional parameter to CMake `-DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda`.

To enable MAGMA (GPU implementation of Lapack) use `-DUSE_MAGMA=On`. Append MAGMA's installation directory to `CMAKE_PREFIX_PATH` if necessary.

#### Parallel eigensolvers
To compile with ScaLAPACK use `-DUSE_SCALAPACK=On`. To use ELPA, both `-DUSE_SCALAPACK=On` and `-DUSE_ELPA=On` are
required, as we need we need ScaLAPACK functionality to transform the generalized eigenvalue problem to standard form,
which can then be solved by ELPA. Append ScaLAPACK's and ELPA's install directory to `CMAKE_PREFIX_PATH` if necessary.

#### Python module
To create Python module you need to specify `-DCREATE_PYTHON_MODULE=On`. SIRIUS Python module depends on `mpi4py` and
`pybind11` packages. They must be installed on your platform.

#### Additional options
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
inside QE is listed below:
```Fortran
subroutine get_band_energies_from_sirius
  !
  use wvfct,    only : nbnd, et
  use klist,    only : nkstot, nks
  use lsda_mod, only : nspin
  use sirius
  !
  implicit none
  !
  integer, external :: global_kpoint_index
  !
  real(8), allocatable :: band_e(:,:)
  integer :: ik, nk, nb, nfv

  allocate(band_e(nbnd, nkstot))

  ! get band energies
  if (nspin.ne.2) then
    ! non-magnetic or non-collinear case
    do ik = 1, nkstot
      call sirius_get_band_energies(ks_handler, ik, 0, band_e(1, ik))
    end do
  else
    ! collinear magnetic case
    nk = nkstot / 2
    ! get band energies
    do ik = 1, nk
      call sirius_get_band_energies(ks_handler, ik, 0, band_e(1, ik))
      call sirius_get_band_energies(ks_handler, ik, 1, band_e(1, nk + ik))
    end do

  endif

  ! convert to Ry
  do ik = 1, nks
    et(:, ik) = 2.d0 * band_e(:, global_kpoint_index(nkstot, ik))
  enddo

  deallocate(band_e)

end subroutine get_band_energies_from_sirius
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

Here is a complete installation of SIRIUS-enabled Quantum ESPRESSO using `nvidia/cuda:10.1-devel-ubuntu18.04`
Linux distribution:

<details><summary>Dockerfile</summary>
<p>

```dockerfile
FROM nvidia/cuda:10.1-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

ENV FORCE_UNSAFE_CONFIGURE 1


ARG SPFFT_VERSION=0.9.10
ARG SIRIUS_VERSION=6.5.2
ARG QE_VERSION=6.5-rc4-sirius
ARG MPICH_VERSION=3.1.4
ENV MPICH_VERSION ${MPICH_VERSION}

RUN apt-get update

RUN apt-get install -y apt-utils

# install basic tools
RUN apt-get install -y gcc g++ gfortran git make unzip \
  vim wget pkg-config python3-pip curl environment-modules tcl \
  apt-transport-https ca-certificates gnupg software-properties-common \
  libhdf5-dev libgsl-dev libxc-dev

## install GCC-8
#RUN apt-get install -y gcc-8 g++-8 gfortran-8
#RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 40
#RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 40
#RUN update-alternatives --install /usr/bin/gfortran gfortran  /usr/bin/gfortran-8 40

# install latest CMake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update
RUN apt-get install -y cmake

# get and build mpich
RUN wget https://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz
RUN tar -xzvf mpich-${MPICH_VERSION}.tar.gz
RUN cd mpich-${MPICH_VERSION} && \
    ./configure && \
    make install -j6
RUN rm mpich-${MPICH_VERSION}.tar.gz
RUN rm -rf mpich-${MPICH_VERSION}

# install MKL
RUN wget -O - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB 2>/dev/null | apt-key add -
RUN apt-add-repository 'deb https://apt.repos.intel.com/mkl all main'

RUN apt-get install -y intel-mkl-2020.0-088

ENV MKLROOT=/opt/intel/compilers_and_libraries/linux/mkl

RUN echo "/opt/intel/lib/intel64 \n/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64 \n/" >> /etc/ld.so.conf.d/intel.conf
RUN ldconfig

WORKDIR /root

# install SpFFT
RUN wget https://github.com/eth-cscs/SpFFT/archive/v$SPFFT_VERSION.tar.gz && tar zxvf v$SPFFT_VERSION.tar.gz

RUN mkdir SpFFT-$SPFFT_VERSION/build && cd SpFFT-$SPFFT_VERSION/build && \
  cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DSPFFT_GPU_BACKEND=CUDA -DSPFFT_SINGLE_PRECISION=ON \
  -DSPFFT_MPI=ON -DSPFFT_OMP=ON -DCMAKE_INSTALL_PREFIX=/usr/local

RUN cd SpFFT-$SPFFT_VERSION/build && make -j12 install

# install SIRIUS
RUN wget https://github.com/electronic-structure/SIRIUS/archive/v$SIRIUS_VERSION.tar.gz && tar zxvf v$SIRIUS_VERSION.tar.gz

RUN cd SIRIUS-$SIRIUS_VERSION && CC=mpicc CXX=mpicxx FC=mpif90 FCCPP=cpp python3 prerequisite.py /usr/local spg

RUN mkdir SIRIUS-$SIRIUS_VERSION/build && cd SIRIUS-$SIRIUS_VERSION/build && LIBSPGROOT=/usr/local \
    cmake .. -DSpFFT_DIR=/usr/local/lib/cmake/SpFFT -DUSE_SCALAPACK=1 -DUSE_MKL=1 -DBUILD_TESTS=1 \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DUSE_CUDA=On -DGPU_MODEL='P100'

RUN cd SIRIUS-$SIRIUS_VERSION/build && make -j12 install

ENV SIRIUS_BINARIES /usr/local/bin/

RUN wget https://github.com/electronic-structure/q-e-sirius/archive/v$QE_VERSION.tar.gz && tar zxvf v$QE_VERSION.tar.gz

RUN cd q-e-sirius-6.5-rc4-sirius && ./configure --with-scalapack --enable-parallel

RUN cd q-e-sirius-6.5-rc4-sirius && \
    sed -i -e "s/^BLAS_LIBS\ *=\ *.*/BLAS_LIBS =/" make.inc && \
    sed -i -e "s/^LAPACK_LIBS\ *=\ *.*/LAPACK_LIBS =/" make.inc && \
    sed -i -e "s/LAPACK_LIBS_SWITCH = internal/LAPACK_LIBS_SWITCH = external/" make.inc && \
    sed -i -e "s/BLAS_LIBS_SWITCH = internal/BLAS_LIBS_SWITCH = external/" make.inc && \
    sed -i -e "s/^DFLAGS\ *=\ *.*/DFLAGS = -D__MPI -D__SCALAPACK -D__DFTI -I\/usr\/local\/include\/sirius /" make.inc && \
    sed -i -e "s/^LD_LIBS\ *=\ *.*/LD_LIBS = -L\/usr\/local\/lib -lsirius -Wl,-rpath,\/usr\/local\/lib -L\$(MKLROOT)\/lib\/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -fopenmp/" make.inc && \
    sed -i -e "s/^FFLAGS\ *=\ *.*/FFLAGS = -march=core-avx2 -O3 -march=haswell -fopenmp -ftree-vectorize -fopt-info -fopt-info-missed -fopt-info-vec -fopt-info-loop /" make.inc && \
    make -j 12 pw

```

</p>
</details>

Once `pw.x` binary is created, you can run it with the same parameters and input file as you run the native QE.
By default, SIRIUS library is not used. To enable SIRIUS pass command-line option `-sirius` to `pw.x`.

```bash
# run in default mode
pw.x -i pw.in
# run with SIRIUS enabled
pw.x -i pw.in -sirius
```

SIRIUS is compiled to use both cuSolver and ELPA eigen-value solvers depending on the number of MPI ranks for band
parallelization. Parallel (CPU only) ELPA eign-solver will be used if the number of MPI ranks for diagonalziation
is a square number (for example, `-ndiag 4`), otherwise sequential cuSolver eigen-solver will be used:

```bash
# use cuSolver solver
pw.x -i pw.in -sirius -ndiag 2
pw.x -i pw.in -sirius -ndiag 3
pw.x -i pw.in -sirius -ndiag 6
...
# use ELPA solver
pw.x -i pw.in -sirius -ndiag 4
pw.x -i pw.in -sirius -ndiag 9
pw.x -i pw.in -sirius -ndiag 16
...
```
In most cases it is more efficient to use sequential GPU eigen-solver, unless your system is sufficiently
large (for example, containing >500 atoms).

SIRIUS library is using OpenMP for node-level parallelization. To run QE/SIRIUS efficiently, follow these simple rules:
 * always prefer k-point pool parallelization over band parallelization
 * use as few MPI ranks as possible for band parallelization
 * by default, use one rank per node and many OMP threads; if the calculated system is really small, try to saturate 
   the GPU card using more MPI ranks (e.g.: on a 12-core node, use 2-3-4 ranks with 6-4-3 OMP threads)

#### Benchmarks
In the following examples we compare performace of native and SIRIUS-enabled versions of QE. CPU-only runs were executed
on the dual-socket multi-core nodes containing two 18-core Intel Broadwell CPUs. GPU rus were executed on the hybrid
nodes containing 12-core Intel Haswell CPU and NVIDIA Tesla P100 card:

|Hybrid partition (Cray XC50)                | Multicore partition (Cray XC40)                  |
|--------------------------------------------|--------------------------------------------------|
|Intel Xeon E5-2690 v3 @2.60GHz, 12 cores <br> NVIDIA Tesla P100 16GB | Two Intel Xeon E5-2695 v4 @2.10GHz (2 x 18 cores)|

Ground state calculation ([input](https://github.com/electronic-structure/benchmarks/tree/master/performance/Si511Ge))
of Si511Ge.

<p align="center">
<img src="doc/images/Si511Ge_perf.png">
</p>

Another example is the variable cell relaxation of B6Ni8 ([input](https://github.com/electronic-structure/benchmarks/tree/master/performance/B6Ni8)).
Brillouin zone contains 204 irreducible k-points and only k-pool parallelization is used.

<p align="center">
<img src="doc/images/B6Ni8_perf.png">
</p>

### CP2K
CP2K code uses SIRIUS library to enable plane-wave functionality. The detailed description of the input parameters
can be found [here](https://manual.cp2k.org) under the `/CP2K_INPUT/FORCE_EVAL/PW_DFT` section.

## Contacts
Have you got any questions, feel free to contact us:
  * Anton Kozhevnikov (anton.kozhevnikov@cscs.ch)
  * Mathieu Taillefumier (mathieu.taillefumier@cscs.ch)
  * Simon Pintarelli (simon.pintarelli@cscs.ch)

## Acknowledgements
The development of SIRIUS library would not be possible without support of the following organizations:
| Logo | Name | URL |
|:----:|:----:|:---:|
|![ethz](doc/images/logo_ethz.png)   | Swiss Federal Institute of Technology in Zürich | https://www.ethz.ch/      |
|![cscs](doc/images/logo_cscs.png)   | Swiss National Supercomputing Centre            | https://www.cscs.ch/      |
|![pasc](doc/images/logo_pasc.png)   | Platform for Advanced Scientific Computing      | https://www.pasc-ch.org/  |
|![pasc](doc/images/logo_marvel.png) | NCCR MARVEL <br> Centre on Computational Design and Discovery of Novel Materials | https://nccr-marvel.ch/ |
|![pasc](doc/images/logo_max.png)    | MAX (MAterials design at the eXascale) <br> European Centre of Excellence | http://www.max-centre.eu/   |
|![pasc](doc/images/logo_prace.png)  | Partnership for Advanced Computing in Europe | https://prace-ri.eu/  |

