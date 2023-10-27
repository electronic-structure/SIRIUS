FROM ubuntu:22.04 as builder

ARG CUDA_ARCH=60

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="$PATH:/spack/bin"

#
#ENV FORCE_UNSAFE_CONFIGURE 1
#
RUN apt-get -y update && apt-get install -y apt-utils

# install basic tools
RUN apt-get install -y gcc g++ gfortran clang libomp-dev libomp-14-dev git make unzip \
  vim wget pkg-config curl tcl m4 cpio automake autoconf apt-transport-https \
  ca-certificates gnupg software-properties-common patchelf meson

RUN apt-get -y upgrade

# install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.7/cmake-3.27.7-linux-x86_64.tar.gz -O cmake.tar.gz && \
    tar zxvf cmake.tar.gz --strip-components=1 -C /usr

# get latest version of spack
RUN git clone https://github.com/spack/spack.git

COPY ./spack /opt/spack
RUN spack repo add --scope system /opt/spack

# set the location of packages built by spack
RUN spack config --scope system add config:install_tree:root:/opt/local
# set cuda_arch for all packages
RUN spack config --scope system add packages:all:variants:cuda_arch=${CUDA_ARCH}
RUN spack config --scope system add packages:all:target:[x86_64]

# find gcc and clang compilers
RUN spack compiler find --scope system
RUN spack external find --all --scope system --not-buildable bash perl sed gcc llvm llvm-doe m4 \
    tar xz bzip2 cpio \
    cmake gmake make ninja meson autoconf automake \
    binutils findutils diffutils coreutils git curl openssh openssl ncurses

RUN spack install mpich@3.4.3

RUN echo $(spack find --format='{prefix.lib}' mpich) > /etc/ld.so.conf.d/mpich.conf
RUN ldconfig

RUN spack install openblas threads=openmp

ENV SPEC_GCC_CPU="sirius@develop %gcc build_type=Release +python +scalapack +vdwxc +fortran +tests +nlcglib +elpa ^openblas threads=openmp ^mpich@3.4.3 ^nlcglib@master ^spfft+single_precision ^umpire~device_alloc target=x86_64"
RUN spack install --fail-fast --only=dependencies $SPEC_GCC_CPU

RUN spack clean --all

#ENV SPEC_GCC_GPU="sirius@develop %gcc build_type=Release +scalapack +vdwxc +fortran +tests +nlcglib +elpa +magma +cuda ^openblas threads=openmp ^mpich@3.4.3 ^nlcglib@master ^spfft+single_precision ^magma+cuda ^umpire+cuda~device_alloc target=x86_64"
#RUN spack install --fail-fast --only=dependencies $SPEC_GCC_GPU
#
#ENV SPEC_CLANG_CPU="sirius@develop %clang build_type=Release +tests ^openblas%gcc threads=openmp ^mpich ~fortran ^spfft+single_precision ^libxc%gcc target=x86_64"
#RUN spack install --fresh --fail-fast --only=dependencies $SPEC_CLANG_CPU
