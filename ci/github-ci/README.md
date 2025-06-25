Dockerfile for the base containers to run github CI workflows.

 * Dockerfile - provides GCC/CUDA/CLANG/MPICH environments

 To build (start from a root folder):
 ```bash
 sudo docker build -t electronicstructure/sirius -f ci/github-ci/Dockerfile .
 ```
