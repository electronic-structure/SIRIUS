FROM ubuntu:22.04

ARG ROCM_ARCH=gfx90a

ENV DEBIAN_FRONTEND=noninteractive

ENV FORCE_UNSAFE_CONFIGURE 1

ENV PATH="/spack/bin:${PATH}"

ENV CMAKE_VERSION=3.26.3

RUN apt-get -y update && apt-get install -y apt-utils

# install basic tools
RUN apt-get install -y gcc g++ gfortran clang git make unzip \
  vim wget pkg-config python3-pip python3-venv curl tcl m4 cpio automake \
  apt-transport-https ca-certificates gnupg software-properties-common \
  patchelf meson liblzma-dev libbz2-dev

# install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz -O cmake.tar.gz && \
    tar zxvf cmake.tar.gz --strip-components=1 -C /usr

# get latest version of spack
RUN git clone -b v0.21.2 https://github.com/spack/spack.git

# add local repo to spack
COPY ./spack /opt/spack
RUN spack repo add --scope system /opt/spack

# set the location of packages built by spack
RUN spack config add config:install_tree:root:/opt/local
# set amdgpu_target for all packages
RUN spack config add packages:all:variants:amdgpu_target=${ROCM_ARCH}
# set basic x86_64 architecture
RUN spack config add packages:all:target:x86_64

# find gcc and clang compilers
RUN spack compiler find
RUN spack external find --all --exclude ncurses
# workaround hip wants to call /usr/bin/llvm-config, but ubuntu renamed it to /usr/bin/llvm-config-14
RUN ln -s /usr/bin/llvm-config-14 /usr/bin/llvm-config

# install big packages
RUN spack install --fail-fast hip%gcc
RUN spack install --fail-fast rocblas%gcc
RUN spack install --fail-fast rocsolver%gcc
RUN spack install --fail-fast hipfft%gcc

RUN spack env create -d /sirius-env-rocm && \
    spack -e /sirius-env-rocm add "sirius@develop %gcc build_type=Release +scalapack +fortran +tests +pugixml +rocm ^openblas ^mpich ^spfft ^umpire+rocm~device_alloc" && \
    spack -e /sirius-env-rocm develop -p /sirius-src sirius@develop && \
    spack -e /sirius-env-rocm install --only=dependencies --fail-fast
