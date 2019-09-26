#!/bin/bash -l
#SBATCH --job-name=sirius-gpup-tests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-core=3
#SBATCH --constraint=gpu
#SBATCH --partition=cscsci
#SBATCH --time=00:20:00
#SBATCH --output=sirius-gpup-tests.out
#SBATCH --error=sirius-gpup-tests.err

set -e

source ${ENVFILE}

(
    export CRAY_CUDA_MPS=1
    export OMP_NUM_THREADS=3
    export MPICH_MAX_THREAD_SAFETY=multiple
    module list
    echo "run-gpup-verification: running on $(hostname)"
    cd ../verification
    ./run_tests_parallel_gpu.x
)
