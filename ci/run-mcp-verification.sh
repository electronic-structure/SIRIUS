#!/bin/bash -l
#SBATCH --job-name=sirius-mcp-tests
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=gpu
#SBATCH --partition=cscsci
#SBATCH --time=00:20:00
#SBATCH --output=sirius-mcp-tests.out
#SBATCH --error=sirius-mcp-tests.err

set -e

source ${ENVFILE}

(
    export OMP_NUM_THREADS=2
    export CRAY_CUDA_MPS=1
    export MPICH_MAX_THREAD_SAFETY=multiple
    module list
    echo "run-mc-parallel-verification: running on $(hostname)"
    cd ../verification
    ./run_tests_parallel.x
)
