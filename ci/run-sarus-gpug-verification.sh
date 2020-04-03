#!/bin/bash -l
#SBATCH --job-name=sirius-sarus-gpup-tests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-core=3
#SBATCH --constraint=gpu
#SBATCH --partition=cscsci
#SBATCH --time=00:20:00
#SBATCH --output=sirius-gpup-tests.out
#SBATCH --error=sirius-gpup-tests.err

set -e

(
    export CRAY_CUDA_MPS=1
    export OMP_NUM_THREADS=3
    export MPICH_MAX_THREAD_SAFETY=multiple
    module list
    echo "run-gpup-verification: running on $(hostname)"
    srun -n4 -c2 --unbuffered --hint=nomultithread --mpi=pmi2 sarus run --mount=type=bind,source=$REPO_FOLDER,destination=$REPO_FOLDER $SARUS_IMAGE bash -c 'cd $REPO_FOLDER/verification/test01 && sirius.scf --test_against=output_ref.json --control.std_evp_solver_name=scalapack --control.gen_evp_solver_name=scalapack --control.mpi_grid_dims=2:2 --control.processing_unit=gpu'

)

