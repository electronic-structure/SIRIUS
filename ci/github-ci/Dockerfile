FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="$PATH:/spack/bin"

#
#ENV FORCE_UNSAFE_CONFIGURE 1
#
RUN apt-get -y update && apt-get install -y apt-utils
RUN apt-get -y upgrade

# install basic tools
RUN apt-get install -y gcc g++ gfortran clang clang-19 lld-19 \
    libomp-dev git make unzip \
    vim wget pkg-config curl tcl m4 cpio automake autoconf apt-transport-https \
    ca-certificates gnupg software-properties-common patchelf \
    file libssl-dev lbzip2

# install CMake
#RUN wget https://github.com/Kitware/CMake/releases/download/v4.0.3/cmake-4.0.3-linux-x86_64.tar.gz -O cmake.tar.gz && \
#    tar zxvf cmake.tar.gz --strip-components=1 -C /usr

# get latest version of spack
RUN git clone -b releases/v1.0 https://github.com/spack/spack.git


RUN git clone https://github.com/spack/spack-packages.git
RUN git -C spack-packages checkout 9b5cdf0c383c9d67ab7c5f271509566e3d2ac758
RUN spack repo set --destination /spack-packages --scope system builtin
#COPY ./spack /opt/spack
#RUN spack repo add --scope system /opt/spack

# set the location of packages built by spack
RUN spack config --scope system add config:install_tree:root:/opt/local
RUN spack config --scope system add packages:all:target:[x86_64]

# find gcc and clang compilers
RUN spack compiler find --scope system

RUN spack external find --all --scope system --not-buildable bash perl sed gcc llvm llvm-doe m4 \
    tar xz bzip2 cpio gmake make autoconf automake \
    binutils findutils diffutils coreutils git openssh openssl ncurses

# create an environment and install only dependencies
RUN spack env create -d /sirius-env
RUN spack -e /sirius-env add sirius@develop build_type=Release +python +scalapack +vdwxc +fortran +tests +elpa +dftd3 +dftd4
RUN spack -e /sirius-env config add packages:all:providers:blas:[openblas]
RUN spack -e /sirius-env config add 'packages:all:require:"%[when=%c,cxxfortran]c=gcc"'
RUN spack -e /sirius-env config add packages:all:providers:lapack:[openblas]
RUN spack -e /sirius-env config add packages:all:providers:mpi:[openmpi]
RUN spack -e /sirius-env config add packages:openblas:require:[threads=openmp]
# RUN spack -e /sirius-env config add packages:spfft:require:[+single_precision]
RUN spack -e /sirius-env config add packages:umpire:require:[~device_alloc]
RUN spack -e /sirius-env concretize -f
RUN spack -e /sirius-env install --only=dependencies


RUN spack env create -d /sirius-env-clang
RUN spack -e /sirius-env-clang add sirius@develop build_type=Release +tests %c,cxx=clang@19
RUN spack -e /sirius-env-clang config add packages:all:providers:blas:[openblas]
RUN spack -e /sirius-env-clang config add 'packages:all:require:"%[when=%c,cxxfortran]c=clang@19"'
RUN spack -e /sirius-env-clang config add packages:all:providers:lapack:[openblas]
RUN spack -e /sirius-env-clang config add packages:all:providers:mpi:[openmpi]
RUN spack -e /sirius-env-clang config add packages:openblas:require:[threads=openmp]
# RUN spack -e /sirius-env config add packages:spfft:require:[+single_precision]
RUN spack -e /sirius-env-clang config add packages:umpire:require:[~device_alloc]
RUN spack -e /sirius-env-clang concretize -f
RUN spack -e /sirius-env-clang install --only=dependencies
