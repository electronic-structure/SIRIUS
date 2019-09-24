#!/bin/bash -l
#SBATCH --job-name=sirius-gpu-tests
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --partition=cscsci
#SBATCH --time=00:20:00
#SBATCH --output=sirius-gpu-tests.out
#SBATCH --error=sirius-gpu-tests.err

set -e

source ${ENVFILE}

(
    export OMP_NUM_THREADS=2
    module list
    echo "run-gpu-verification: running on $(hostname)"
    cd ../verification
    ./run_tests_gpu.x
)
