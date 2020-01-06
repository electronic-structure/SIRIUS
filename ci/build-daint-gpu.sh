#!/bin/bash -l
#SBATCH --job-name=build-daint-gpu
#SBATCH --nodes=1
#SBATCH --constraint=mc
#SBATCH --partition=cscsci
#SBATCH --time=00:15:00
#SBATCH --output=build-daint-gpu.out
#SBATCH --error=build-daint-gpu.err

set -e

source ${ENVFILE}

mkdir -p build
(
    cd build
    CXX=CC CC=cc cmake -DUSE_MKL=On \
          -DCMAKE_CXX_FLAGS="-Werror" \
          -DUSE_ELPA=On \
          -DGPU_MODEL=P100 \
          -DUSE_SCALAPACK=On \
          -DUSE_MAGMA=On \
          -DUSE_CUDA=On \
          -DCMAKE_BUILD_TYPE=RELWITHDEBINFO \
          -DCREATE_PYTHON_MODULE=On \
          ../
    make clean
    make -j VERBOSE=1
) && echo "build successful"
