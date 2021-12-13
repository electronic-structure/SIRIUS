# This is mainly to provide CUDA, so you can extend e.g.
# - nvidia/cuda:10.2-devel-ubuntu18.04
# - nvidia/cuda:11.0-devel-ubuntu20.04
# - ubuntu:20.04
# e.g. --build-arg BASE_IMAGE=nvidia/cuda:10.2-devel-ubuntu18.04
ARG BASE_IMAGE=ubuntu:20.04

FROM $BASE_IMAGE

# Set this to a spack.yaml file which contains a spec
# e.g. --build-arg SPACK_ENVIRONMENT=ci/spack/my-env.yaml
ARG SPACK_ENVIRONMENT
ARG SPACK_SHA
ARG COMPILER_CONFIG
ENV DEBIAN_FRONTEND=noninteractive \
    PATH="$PATH:/opt/spack/bin:/opt/libtree" \
    SPACK_COLOR=always
SHELL ["/bin/bash", "-c"]

RUN apt-get -yqq update \
 && apt-get -yqq install --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        curl \
        file \
        g++ \
        gcc \
        gfortran \
        git \
        gnupg2 \
        iproute2 \
        jq \
        libomp-dev \
        lmod \
        locales \
        lua-posix \
        make \
        parallel \
        patchelf \
        python3 \
        tar \
        tcl \
        unzip \
 && locale-gen en_US.UTF-8 \
 && rm -rf /var/lib/apt/lists/*

# Install libtree for packaging
RUN mkdir -p /opt/libtree && \
    curl -Lfso /opt/libtree/libtree https://github.com/haampie/libtree/releases/download/v2.0.0/libtree_x86_64 && \
    chmod +x /opt/libtree/libtree

# This is the spack version we want to have
ENV SPACK_SHA=$SPACK_SHA

# Install the specific ref of Spack provided by the user
RUN mkdir -p /opt/spack && \
    curl -Ls "https://api.github.com/repos/spack/spack/tarball/$SPACK_SHA" | tar --strip-components=1 -xz -C /opt/spack

# "Install" compilers
COPY "$COMPILER_CONFIG" /opt/spack/etc/spack/compilers.yaml

# Set up the binary cache and trust the public part of our signing key
COPY ./ci/spack/public_key.asc ./public_key.asc
RUN spack mirror add --scope site cscs https://spack.cloud && \
    spack gpg trust ./public_key.asc

# Add our custom spack repo from here
COPY ./spack /user_repo

RUN spack repo add --scope site /user_repo

# Build dependencies
# 1. Create a spack environment named `ci` from the input spack.yaml file
# 2. Install only the dependencies of this (top level is our package)
COPY $SPACK_ENVIRONMENT /spack_environment/spack.yaml
RUN spack env create --without-view ci /spack_environment/spack.yaml
RUN spack -e ci install --fail-fast --only=dependencies --require-full-hash-match
