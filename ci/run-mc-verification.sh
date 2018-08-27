#!/bin/bash -l
#SBATCH --job-name=sirius-mc-tests
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --partition=cscsci
#SBATCH --time=00:20:00
#SBATCH --output=sirius-mc-tests.out
#SBATCH --error=sirius-mc-tests.err

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

loadDependencies

(
    module list
    echo "run-mc-verification: running on $(hostname)"
    cd ../verification
    ./run_tests.x
)
