FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG CUDA_ARCH=60

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="$PATH:/spack/bin"

#
#ENV FORCE_UNSAFE_CONFIGURE 1
#
RUN apt-get -y update && apt-get install -y apt-utils

# install basic tools
RUN apt-get install -y gcc g++ gfortran clang libomp-dev libomp-14-dev git make unzip \
  vim wget pkg-config python3-pip curl tcl m4 cpio automake autoconf \
  apt-transport-https ca-certificates gnupg software-properties-common \
  patchelf meson cython3 python3-pythran

# install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.3/cmake-3.26.3-linux-x86_64.tar.gz -O cmake.tar.gz && \
    tar zxvf cmake.tar.gz --strip-components=1 -C /usr

# get latest version of spack
RUN git clone https://github.com/spack/spack.git

COPY ./spack /opt/spack
RUN spack repo add --scope system /opt/spack

# set the location of packages built by spack
RUN spack config --scope system add config:install_tree:root:/opt/local
# set cuda_arch for all packages
RUN spack config --scope system add packages:all:variants:cuda_arch=${CUDA_ARCH}
RUN spack config --scope system add packages:all:target:x86_64

# find gcc and clang compilers
RUN spack compiler find --scope system
RUN spack external find --all --scope system

## install yq (utility to manipulate the yaml files)
#RUN wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_386 && chmod a+x /usr/local/bin/yq
#
## change the fortran compilers: for gcc the gfortran is already properly set and the change has no effect; add it for clang
#RUN yq -i '.compilers[0].compiler.paths.f77 = "/usr/bin/gfortran"' /etc/spack/compilers.yaml && \
#    yq -i '.compilers[0].compiler.paths.fc = "/usr/bin/gfortran"' /etc/spack/compilers.yaml  && \
#    yq -i '.compilers[1].compiler.paths.f77 = "/usr/bin/gfortran"' /etc/spack/compilers.yaml && \
#    yq -i '.compilers[1].compiler.paths.fc = "/usr/bin/gfortran"' /etc/spack/compilers.yaml

ENV SPEC_GCC_CPU="sirius@develop %gcc build_type=Release +python +scalapack +vdwxc +fortran +tests +nlcglib +elpa ^openblas ^mpich@3.4.3 ^nlcglib +wrapper ^kokkos@3.7.01 +wrapper ^spfft+single_precision ^umpire~device_alloc target=x86_64"
RUN spack install --fail-fast --only=dependencies $SPEC_GCC_CPU

ENV SPEC_GCC_GPU="sirius@develop %gcc build_type=Release +python +scalapack +vdwxc +fortran +tests +nlcglib +elpa +magma +cuda ^openblas ^mpich@3.4.3 ^nlcglib +cuda +wrapper ^kokkos@3.7.01 +wrapper ^spfft+single_precision ^magma+cuda ^umpire+cuda~device_alloc target=x86_64"
RUN spack install --fail-fast --only=dependencies $SPEC_GCC_GPU

ENV SPEC_CLANG_CPU="sirius@develop %clang build_type=Release +tests ^openblas%gcc ^mpich ~fortran ^spfft+single_precision ^libxc%gcc target=x86_64"
RUN spack install --fresh --fail-fast --only=dependencies $SPEC_CLANG_CPU
