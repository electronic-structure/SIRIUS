Dockerfiles for the base containers that are used to run github CI workflows.


 * Dockerfile - provides GCC/CUDA/CLANG environments
 * Dockerfile.rocm - provides GCC/ROCm environment

 Beware: it takes ~6H on a modern workstation to rebuild ROCm container!

 To build:
 ```bash
 sudo docker build -t electronicstructure/sirius -f Dockerfile .
 ```
