#FROM nvidia/cuda:11.3.0-runtime-ubuntu20.04
FROM ubuntu:20.04

#==ENV NCCL_VERSION 2.9.6
#==
ENV DEBIAN_FRONTEND noninteractive
#==
ENV FORCE_UNSAFE_CONFIGURE 1

RUN apt-get update && apt-get install -y --no-install-recommends \
  apt-utils python3-distutils python3-apt python3-dev \
  gcc g++ gfortran python3 git make apt-transport-https ca-certificates \
  gnupg software-properties-common curl patch pkg-config vim wget xz-utils unzip
#==
#==RUN apt-get update && apt-get install -y --no-install-recommends \
#==    libtinfo5 libncursesw5 \
#==    gcc g++ gfortran git make \
#==    cuda-cudart-dev-11-3=11.3.58-1 \
#==    cuda-command-line-tools-11-3=11.3.0-1 \
#==    cuda-minimal-build-11-3=11.3.0-1 \
#==    cuda-libraries-dev-11-3=11.3.0-1 \
#==    cuda-nvml-dev-11-3=11.3.58-1 \
#==    libnpp-dev-11-3=11.3.3.44-1 \
#==    libnccl-dev=2.9.6-1+cuda11.3 \
#==    libcublas-dev-11-3=11.4.2.10064-1 \
#==    libcusparse-dev-11-3=11.5.0.58-1 \
#==    python \
#==    && rm -rf /var/lib/apt/lists/*
#==
#==# apt from auto upgrading the cublas package. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
#==RUN apt-mark hold libcublas-dev-11-3 libnccl-dev
#==ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
#==
# get latest version of spack
#RUN git clone https://github.com/spack/spack.git /usr/spack
#
#ENV SPACK_ROOT /usr/spack
#
#ENV PATH="${SPACK_ROOT}/bin:${PATH}"
#
#COPY compilers.yaml /root/.spack/linux/

# install spack to /spack/ folder
RUN wget -qO spack.x https://github.com/haampie/spack-batteries-included/releases/download/develop/spack-x86_64.x
RUN chmod +x spack.x
RUN ./spack.x --squashfs-extract
# add environment variables
RUN echo "source /spack/spack_src/share/spack/setup-env.sh" >> /etc/profile.d/spack.sh

RUN /spack/spack install sirius@develop +fortran +scalapack

#COPY test.f90 /root/test.f90

#==#
#==#RUN /bin/bash -c eval `spack load --sh pkgconf`
#==#
#==
#==
#==
#==
#==# RUN apt-get install -y apt-utils
#==# 
#==# # install basic tools
#==# RUN apt-get install -y gcc g++ gfortran git make cmake unzip \
#==#   vim wget pkg-config python3-pip curl environment-modules tcl
#==# 
#==# # add environment variables
#==# RUN echo "source /root/spack/share/spack/setup-env.sh" >> /etc/profile.d/spack.sh
#==# 
#==# ## build GCC
#==# #RUN /bin/bash -l -c "spack install gcc@8.2.0"
#==# #
#==# ## add GCC to environment
#==# #RUN echo "spack load --dependencies gcc@8.2.0" >> /etc/profile.d/spack.sh
#==# #
#==# ## update list of spack compilers
#==# #RUN /bin/bash -l -c "spack compiler find"
#==# #
#==# ## build CMake
#==# #RUN /bin/bash -l -c "spack install --no-checksum cmake@3.16.2 %gcc@8.2.0"
#==# #
WORKDIR /root

ENTRYPOINT ["bash", "-l"]
