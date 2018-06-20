configure.py/Makefile
*********************

For example: ``python configure.py platforms/platform.Linux.GNU.json``

CMake on daint.cscs.ch
**********************

Install libxc, GSL, spglib, magma via easybuild.

1) load modules

  .. code-block:: bash

      module load git
      module load daint-gpu
      module load EasyBuild-custom/cscs
      module swap PrgEnv-cray PrgEnv-intel
      module unload cray-libsci
      module load cray-hdf5
      module load cudatoolkit
      module load CMake/3.8.1
      module load libxc
      module load GSL/2.4-CrayIntel-17.08
      module load spglib/1.10.3-CrayIntel-17.08
      module load magma/2.3.0-CrayIntel-17.08-cuda-8.0
      module load intel
      module load gcc

2) run cmake

  .. code-block:: bash

      git clone https://github.com/electronic-structure/SIRIUS.git -b develop code
      mkdir -p build
      cd build
      cmake -DCMAKE_INSTALL_PREFIX=${HOME}/local/sirius \
            -DUSE_MKL=On \
            -DUSE_ELPA=Off \
            -DCREATE_PYTHON_MODULE=Off \
            -DUSE_MAGMA=On \
            -DUSE_CUDA=On \
            ../code

      make ${MAKEFLAGS} install
