#!/bin/bash

set -xeuo pipefail

export SPACK_SYSTEM_CONFIG_PATH=/user-environment/config

spack --version

# make sure we keep the stage direcorty
spack config --scope=user add config:build_stage:/dev/shm/spack-stage

spack env create -d ./spack-env
# add local repository with current sirius recipe
spack -e ./spack-env repo add $REPO

spack -e ./spack-env config add "packages:all:variants:[cuda_arch=${CUDA_ARCH},+cuda]"

# workaround, first command fails asking to update config format, doesn't make any sense, cannot reproduce on cli
#spack -e ./spack-env config add config:install_tree:$SPACK_INSTALL_TREE
yq w -i ./spack-env/spack.yaml 'spack.config.install_tree.root' $SPACK_INSTALL_TREE
#spack -e ./spack-env config add 'config:install_tree:projections:all:"{name}-{version}"'
yq w -i ./spack-env/spack.yaml 'spack.config.install_tree.projections.all' '{name}-{version}'
# debug

spack -e ./spack-env add $SPEC

# build sirius from source
spack -e ./spack-env develop -p $PWD sirius@develop

# display spack.yaml
cat ./spack-env/spack.yaml

spack -e ./spack-env concretize
spack -e ./spack-env install

# the tar pipe below expects a relative path
builddir=$(spack -e ./spack-env location -b sirius)
# create a symlink to spack build directory (keep in artifacts)
tar -cf builddir.tar $builddir
