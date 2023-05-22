FROM ubuntu:22.04 as builder

ARG CUDA_ARCH=60

ENV DEBIAN_FRONTEND noninteractive

ENV FORCE_UNSAFE_CONFIGURE 1

ENV PATH="/spack/bin:${PATH}"

ENV MPICH_VERSION=3.4.3

ENV CMAKE_VERSION=3.26.3

RUN apt-get -y update

RUN apt-get install -y apt-utils

# install basic tools
RUN apt-get install -y --no-install-recommends gcc g++ gfortran clang libomp-14-dev git make unzip file \
  vim wget pkg-config python3-pip python3-dev cython3 python3-pythran curl tcl m4 cpio automake meson \
  xz-utils patch patchelf apt-transport-https ca-certificates gnupg software-properties-common perl tar bzip2

# install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz -O cmake.tar.gz && \
    tar zxvf cmake.tar.gz --strip-components=1 -C /usr

# get latest version of spack
RUN git clone https://github.com/spack/spack.git

# set the location of packages built by spack
RUN spack config add config:install_tree:root:/opt/local
# set cuda_arch for all packages
RUN spack config add packages:all:variants:cuda_arch=${CUDA_ARCH}

# find all external packages
RUN spack external find --all

# find compilers
RUN spack compiler find

# install yq (utility to manipulate the yaml files)
RUN wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_386 && chmod a+x /usr/local/bin/yq

# change the fortran compilers: for gcc the gfortran is already properly set and the change has no effect; add it for clang
RUN yq -i '.compilers[0].compiler.paths.f77 = "/usr/bin/gfortran"' /root/.spack/linux/compilers.yaml && \
    yq -i '.compilers[0].compiler.paths.fc = "/usr/bin/gfortran"' /root/.spack/linux/compilers.yaml  && \
    yq -i '.compilers[1].compiler.paths.f77 = "/usr/bin/gfortran"' /root/.spack/linux/compilers.yaml && \
    yq -i '.compilers[1].compiler.paths.fc = "/usr/bin/gfortran"' /root/.spack/linux/compilers.yaml

# install MPICH
RUN spack install --only=dependencies mpich@${MPICH_VERSION} %gcc
RUN spack install mpich@${MPICH_VERSION} %gcc

# install openmpi
RUN spack install --only=dependencies openmpi %gcc
RUN spack install openmpi %gcc

# install libvdwxc
RUN spack install libvdwxc %gcc +mpi ^mpich@${MPICH_VERSION}

# install openblas
RUN spack install openblas %gcc +fortran

RUN spack install magma %gcc +cuda +fortran ^openblas

RUN spack install nlcglib %gcc +cuda+wrapper ^kokkos@3.7.01+wrapper

# for the MPI hook
RUN echo $(spack find --format='{prefix.lib}' mpich) > /etc/ld.so.conf.d/mpich.conf
RUN ldconfig

# install dependencies of several basic configurations

RUN spack install --only=dependencies --fail-fast \
  "sirius@develop %gcc build_type=RelWithDebInfo +fortran +tests +apps +cuda +scalapack ^mpich ^spfft+single_precision+cuda ^intel-oneapi-mkl+cluster ^umpire+cuda~device_alloc"

RUN spack install --only=dependencies --fail-fast \
  "sirius@develop %gcc build_type=RelWithDebInfo +tests +apps +cuda +scalapack +elpa ^mpich ^intel-oneapi-mkl+cluster ^elpa+cuda ^spfft+single_precision+cuda ^umpire+cuda~device_alloc"

RUN spack install --only=dependencies --fail-fast \
  "sirius@develop %gcc build_type=RelWithDebInfo +tests +apps +cuda +magma ^mpich ^openblas ^magma+cuda ^umpire+cuda~device_alloc"

RUN spack install --only=dependencies --fail-fast \
  "sirius@develop %gcc build_type=RelWithDebInfo +fortran +tests +apps +cuda +scalapack +single_precision ^mpich ^intel-oneapi-mkl+cluster ^umpire+cuda~device_alloc"

RUN spack install --only=dependencies --fail-fast \
  "sirius@develop %gcc build_type=RelWithDebInfo +fortran +tests +apps +cuda +scalapack +vdwxc ^mpich ^openblas ^umpire+cuda~device_alloc"

RUN spack install --only=dependencies --fail-fast \
  "sirius@develop %gcc build_type=RelWithDebInfo +fortran +tests +apps +vdwxc +cuda +nlcglib ^openblas ^mpich ^nlcglib +cuda +wrapper ^kokkos +wrapper ^umpire+cuda~device_alloc"

RUN spack install --only=dependencies --fail-fast \
  "sirius@develop %gcc +tests +apps +scalapack +fortran build_type=RelWithDebInfo ^openblas ^openmpi ^umpire~cuda~device_alloc"

RUN spack install --only=dependencies --fail-fast \
  "sirius@develop %clang build_type=RelWithDebInfo ~fortran +tests ^openblas%gcc ^libxc%gcc ^mpich%gcc ^umpire~cuda~device_alloc"
