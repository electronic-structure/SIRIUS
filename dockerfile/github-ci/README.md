Dockerfiles for the base containers that are used to run github CI workflows.


 * Dockerfile - provides GCC/CUDA/CLANG/MPICH environments
 * Dockerfile.rocm - provides GCC/ROCm/MPICH environment
 * Dockerfile.openmpi - provides GCC/CUDA/OpenMPI environment

 Beware: it takes ~6H on a modern workstation to rebuild ROCm container!

 To build:
 ```bash
 sudo docker build -t electronicstructure/sirius -f Dockerfile .
 ```
