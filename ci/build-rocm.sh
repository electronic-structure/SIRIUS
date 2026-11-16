#!/bin/bash

set -xeuo pipefail

# make sure we keep the stage direcorty
spack config --scope=spack add config:build_stage:/dev/shm/spack-stage
# we might need to install dependencies too, e.g. nlcglib in case of API changes
spack config --scope=spack add config:install_tree:root:/dev/shm/spack-stage

spack env create -d ./spack-env
# add local repository with current sirius recipe
# spack -e ./spack-env repo add $REPO
spack -e ./spack-env config add 'include:[/user-environment/config]'

spack -e ./spack-env config add "packages:all:variants:[amdgpu_target=${ROCM_ARCH},amdgpu_target_sram_ecc=${ROCM_ARCH},+rocm]"
# TODO: update once this is has changed upstream
# spack -e ./spack-env config add "packages:all:providers:mpi:[cray-mpich@9.1.0+rocm]"

spack -e ./spack-env add $SPEC
# pytest is needed to run the python module tests (ctest -L python)
if [[ "$SPEC" == *+python* ]]; then
  spack -e ./spack-env add py-pytest
fi
spack -e ./spack-env add cray-mpich@9.1.0+rocm

# build sirius from source
spack -e ./spack-env develop -p $PWD sirius@develop

# display spack.yaml
cat ./spack-env/spack.yaml

spack -e ./spack-env concretize
spack -e ./spack-env install

# runtime environment for the python tests (PYTHONPATH etc.), sourced by the CI test jobs
if [[ "$SPEC" == *+python* ]]; then
  spack -e ./spack-env load --sh sirius py-pytest > /dev/shm/spack-stage/sirius-python-env.sh
fi

# the tar pipe below expects a relative path
builddir=$(spack -e ./spack-env location -b sirius)
# create a symlink to spack build directory (keep in artifacts)
extra=()
[[ -f /dev/shm/spack-stage/sirius-python-env.sh ]] && extra+=(/dev/shm/spack-stage/sirius-python-env.sh)
tar -cf builddir.tar $builddir "${extra[@]}"
