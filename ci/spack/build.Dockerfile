# This is mainly to provide CUDA, so you can extend e.g.
# - nvidia/cuda:10.2-devel-ubuntu18.04
# - nvidia/cuda:11.0-devel-ubuntu20.04
# - ubuntu:18.04
# e.g. --build-arg BASE_IMAGE=nvidia/cuda:10.2-devel-ubuntu18.04
ARG BASE_IMAGE=ubuntu:18.04

FROM $BASE_IMAGE

# Set this to a spack.yaml file which contains a spec
# e.g. --build-arg SPACK_ENVIRONMENT=ci/spack/my-env.yaml
ARG SPACK_ENVIRONMENT

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="$PATH:/opt/spack/bin"

SHELL ["/bin/bash", "-c"]

RUN apt-get -yqq update \
 && apt-get -yqq install --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        file \
        g++ \
        gcc \
        gfortran \
        git \
        gnupg2 \
        iproute2 \
        lmod \
        locales \
        lua-posix \
        make \
        parallel \
        python3 \
        python3-pip \
        python3-setuptools \
        tcl \
        unzip \
 && locale-gen en_US.UTF-8 \
 && pip3 install boto3 \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/local/cmake && \
    curl -Ls https://github.com/Kitware/CMake/releases/download/v3.18.0/cmake-3.18.0-Linux-x86_64.tar.gz | tar --strip-components=1 -xz -C /usr/local/cmake


# This is the spack version we want to have
ARG SPACK_SHA
ENV SPACK_SHA=$SPACK_SHA

# Install the specific ref of Spack provided by the user
RUN mkdir -p /opt/spack && \
    curl -Ls "https://api.github.com/repos/spack/spack/tarball/$SPACK_SHA" | tar --strip-components=1 -xz -C /opt/spack

# Add our custom spack repo from here
COPY ./spack /user_repo

RUN spack repo add --scope system /user_repo

# Set up the binary cache and trust the public part of our signing key
COPY ./ci/spack/public_key.asc ./public_key.asc
RUN spack mirror add --scope system minio http://148.187.98.133:9000/spack && \
    spack gpg trust ./public_key.asc

# Copy over the environment file
COPY $SPACK_ENVIRONMENT /spack_environment/spack.yaml

# Build dependencies
# 1. Create a spack environment named `ci` from the input spack.yaml file
# 2. Install only the dependencies of this (top level is our package)
RUN spack --color=always env create --without-view ci /spack_environment/spack.yaml

RUN spack --color=always -e ci install --only=dependencies
