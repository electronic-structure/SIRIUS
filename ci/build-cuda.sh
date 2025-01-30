#!/bin/bash

set -xeuo pipefail

export SPACK_SYSTEM_CONFIG_PATH=/user-environment/config

# make sure we keep the stage direcorty
spack config --scope=user add config:build_stage:$PWD


spack env create -d ./spack-env
# add local repository with current sirius recipe
spack -e ./spack-env repo add ./ci/sirius-uenv-recipe/repo

spack -e ./spack-env config add "packages:all:variants:[cuda_arch=${CUDA_ARCH},+cuda]"

spack -e ./spack-env add $SPEC

# build sirius from source
spack -e ./spack-env develop -p $PWD sirius@develop

# display spack.yaml
cat ./spack-env/spack.yaml

spack -e ./spack-env concretize
spack -e ./spack-env install

builddir=$(spack -e ./spack-env location -b sirius)
# create a symlink to spack build directory (keep in artifacts)
mkdir builddir
tar --dereference -cf - $builddir | tar --strip-components=1 - -C builddir xf
