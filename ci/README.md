Dockerfiles to run CI/CD workflows.

 * baseimage.cuda.Dockerfile - base image for CSCS CI/CD workflow
 * baseimage.github.Dockerfile - provides GCC/CUDA/CLANG/MPICH environment for testing on github

 To build:
 ```bash
 sudo docker build -t electronicstructure/sirius -f baseimage.github.Dockerfile .
 ```
