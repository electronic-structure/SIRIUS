#!/bin/bash -l
#SBATCH --job-name=build-daint-gpu
#SBATCH --nodes=1
#SBATCH --constraint=mc
#SBATCH --partition=cscsci
#SBATCH --time=00:15:00
#SBATCH --output=build-daint-gpu.out
#SBATCH --error=build-daint-gpu.err

set -e

loadDependencies()
{
    module purge
    export EASYBUILD_PREFIX=/users/simonpi/jenkins
    module load modules
    module load craype
    module load PrgEnv-gnu
    module load daint-gpu
    module unload cray-libsci
    module load EasyBuild-custom/cscs
    module load cray-hdf5
    module load cudatoolkit
    module load CMake/3.8.1
    module load intel
    module load gcc
    module load cray-python/3.6.1.1
    module load git

    module load libxc/4.2.3-CrayGNU-17.08
    module load GSL/2.4-CrayGNU-17.08
    module load spglib/1.10.3-CrayGNU-17.08
    module load magma/2.3.0-CrayGNU-17.08-cuda-8.0
}

#compile easybuild modules and load env
loadDependencies

mkdir -p build
(
    cd build
    cmake -DUSE_MKL=On -DUSE_ELPA=Off \
          -DGPU_MODEL=P100 \
          -DUSE_MAGMA=On -DUSE_CUDA=On \
          ../
    make -j24 VERBOSE=1
) && echo "build successful"
