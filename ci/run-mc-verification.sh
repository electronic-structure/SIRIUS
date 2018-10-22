#!/bin/bash -l
#SBATCH --job-name=sirius-mc-tests
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --partition=cscsci
#SBATCH --time=00:20:00
#SBATCH --output=sirius-mc-tests.out
#SBATCH --error=sirius-mc-tests.err

set -e

source ${ENVFILE}

(
    export OMP_NUM_THREADS=2
    module list
    echo "run-mc-verification: running on $(hostname)"
    cd ../verification
    ./run_tests.x
)
