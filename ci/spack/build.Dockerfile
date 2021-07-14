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

# Compiler.yaml file for spack
ARG COMPILER_CONFIG

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="$PATH:/opt/spack/bin:/opt/libtree"

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
        python3 \
        tar \
        tcl \
        unzip \
 && locale-gen en_US.UTF-8 \
 && rm -rf /var/lib/apt/lists/*

# Install libtree for packaging
RUN mkdir -p /opt/libtree && \
    curl -Ls https://github.com/haampie/libtree/releases/download/v1.2.0/libtree_x86_64.tar.gz | tar --strip-components=1 -xz -C /opt/libtree

# This is the spack version we want to have
ARG SPACK_SHA
ENV SPACK_SHA=$SPACK_SHA

# Install the specific ref of Spack provided by the user
RUN mkdir -p /opt/spack && \
    curl -Ls "https://api.github.com/repos/spack/spack/tarball/$SPACK_SHA" | tar --strip-components=1 -xz -C /opt/spack

# "Install" compilers
COPY "$COMPILER_CONFIG" /opt/spack/etc/spack/compilers.yaml

# Set up the binary cache and trust the public part of our signing key
COPY ./ci/spack/public_key.asc ./public_key.asc
RUN spack mirror add --scope site cscs https://spack.dev && \
    spack gpg trust ./public_key.asc

# Install clingo and use the new concretizer by default (temporarily until this is the default in spack v0.17)
RUN spack env create -d /clingo                                        && \
    spack -e /clingo add clingo@spack build_type=Release target=x86_64 && \
    spack -e /clingo add py-boto3 target=x86_64                        && \
    spack -e /clingo install --require-full-hash-match                 && \
    echo "config:"                        >> /opt/spack/etc/spack/config.yaml && \
    echo "  concretizer: clingo"          >> /opt/spack/etc/spack/config.yaml

ENV PATH="/clingo/.spack-env/view/bin:$PATH"

# Add our custom spack repo from here
COPY ./spack /user_repo

RUN spack repo add --scope site /user_repo

# Copy over the environment file
COPY $SPACK_ENVIRONMENT /spack_environment/spack.yaml

# Build dependencies
# 1. Create a spack environment named `ci` from the input spack.yaml file
# 2. Install only the dependencies of this (top level is our package)
RUN spack --color=always env create --without-view ci /spack_environment/spack.yaml

RUN spack --color=always -e ci install --fail-fast --only=dependencies --require-full-hash-match
