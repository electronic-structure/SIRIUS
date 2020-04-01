FROM nvidia/cuda:10.1-devel-ubuntu18.04

WORKDIR /root

ENV DEBIAN_FRONTEND noninteractive

ENV FORCE_UNSAFE_CONFIGURE 1

ARG MPICH_VERSION=3.1.4
ARG SPFFT_VERSION=0.9.10

ENV MPICH_VERSION ${MPICH_VERSION}

# install basic tools
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils gcc g++ gfortran git make unzip \
  vim wget pkg-config python3-pip curl environment-modules tcl \
  apt-transport-https ca-certificates gnupg software-properties-common \
  libhdf5-dev libgsl-dev && rm -rf /var/lib/apt/lists/*

RUN wget -qO- "https://cmake.org/files/v3.17/cmake-3.17.0-Linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local

RUN wget https://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz && \
    tar -xzvf mpich-${MPICH_VERSION}.tar.gz && \
    cd mpich-${MPICH_VERSION} && \
    ./configure && \
    make install -j$(nproc) && \
    rm -rf /root/mpich-${MPICH_VERSION}.tar.gz /root/mpich-${MPICH_VERSION}

# install MKL
RUN wget -O - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB 2>/dev/null | apt-key add -

RUN apt-add-repository 'deb https://apt.repos.intel.com/mkl all main'

RUN apt-get install -y --no-install-recommends intel-mkl-2020.0-088

ENV MKLROOT=/opt/intel/compilers_and_libraries/linux/mkl

RUN echo "/opt/intel/lib/intel64 \n/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64 \n/opt/intel/impi/2019.1.144/intel64/lib/release \n/opt/intel/impi/2019.1.144/intel64/lib/ \n/opt/intel/impi/2019.1.144/intel64/libfabric/lib/prov/ \n/opt/intel/impi/2019.1.144/intel64/libfabric/lib/" >> /etc/ld.so.conf.d/intel.conf

RUN ldconfig

# install SpFFT
RUN wget https://github.com/eth-cscs/SpFFT/archive/v$SPFFT_VERSION.tar.gz && \
    tar zxvf v$SPFFT_VERSION.tar.gz && \ 
    mkdir SpFFT-$SPFFT_VERSION/build && \
    cd SpFFT-$SPFFT_VERSION/build && \
    cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DSPFFT_GPU_BACKEND=CUDA -DSPFFT_SINGLE_PRECISION=ON \
             -DSPFFT_MPI=ON -DSPFFT_OMP=ON -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) install && \
    rm -rf /root/v$SPFFT_VERSION.tar.gz

# install SIRIUS
COPY . /build-sirius

RUN mkdir /build-sirius/build && cd /build-sirius/build && CC=mpicc CXX=mpicxx FC=mpif90 FCCPP=cpp python3 ../prerequisite.py /usr/local xc spg && LIBXCROOT=/usr/local LIBSPGROOT=/usr/local \
    cmake .. -DSpFFT_DIR=/usr/local/lib/cmake/SpFFT -DUSE_SCALAPACK=1 -DUSE_MKL=1 -DBUILD_TESTS=1 \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DUSE_CUDA=On -DGPU_MODEL='P100' && make -j$(nproc) DESTDIR=/root/SIRIUS.AppDir install && rm -rf /build-sirius

RUN ldconfig
