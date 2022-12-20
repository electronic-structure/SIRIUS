
Base container provides all dependencies.

sudo docker build -t sirius_base --build-arg CUDA_ARCH=80 .
sudo docker run -it sirius_base bash

