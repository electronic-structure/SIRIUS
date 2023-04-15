FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="$PATH:/spack/bin"

SHELL ["/bin/bash", "-c"]

#
#ENV FORCE_UNSAFE_CONFIGURE 1
#
RUN apt-get -y update && apt-get install -y apt-utils

# install basic tools
RUN apt-get install -y gcc g++ gfortran clang git make unzip \
  vim wget pkg-config python3-pip curl tcl m4 cpio automake \
  apt-transport-https ca-certificates gnupg software-properties-common \
  patchelf

# install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.3/cmake-3.24.3-linux-x86_64.tar.gz -O cmake.tar.gz && \
    tar zxvf cmake.tar.gz --strip-components=1 -C /usr

# get latest version of spack
RUN git clone https://github.com/spack/spack.git

# find gcc and clang compilers
RUN spack compiler find
RUN spack external find --all

# install yq (utility to manipulate the yaml files)
RUN wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_386 && chmod a+x /usr/local/bin/yq

# change the fortran compilers: for gcc the gfortran is already properly set and the change has no effect; add it for clang
RUN yq -i '.compilers[0].compiler.paths.f77 = "/usr/bin/gfortran"' /root/.spack/linux/compilers.yaml && \
    yq -i '.compilers[0].compiler.paths.fc = "/usr/bin/gfortran"' /root/.spack/linux/compilers.yaml  && \
    yq -i '.compilers[1].compiler.paths.f77 = "/usr/bin/gfortran"' /root/.spack/linux/compilers.yaml && \
    yq -i '.compilers[1].compiler.paths.fc = "/usr/bin/gfortran"' /root/.spack/linux/compilers.yaml

# create GCC build env.
RUN spack env create --with-view /sirius-install gcc-build-env

ENV SPEC_GCC="sirius@develop %gcc build_type=Release +python +scalapack +vdwxc +fortran +tests +nlcglib +elpa +magma +cuda cuda_arch=60 ^openblas ^mpich@3.3.2 ^nlcglib +cuda +wrapper cuda_arch=60 ^kokkos +wrapper cuda_arch=60 ^spfft+single_precision ^magma+cuda cuda_arch=60 target=x86_64"

# show spec.
RUN spack -e gcc-build-env spec $SPEC_GCC

RUN spack -e gcc-build-env add $SPEC_GCC

RUN spack -e gcc-build-env install --fail-fast --only=dependencies

# create clang build env.
RUN spack env create clang-build-env

ENV SPEC_CLANG="sirius@develop %clang build_type=Release +tests ^openblas ^mpich ~fortran ^spfft+single_precision target=x86_64"

RUN spack -e clang-build-env spec $SPEC_CLANG

# install all packages; don't reuse from gcc
RUN spack -e clang-build-env add $SPEC_CLANG
RUN spack -e clang-build-env install --fresh --fail-fast --only=dependencies

