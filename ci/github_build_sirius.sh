#!/bin/bash -e

export SPEC="sirius@develop %gcc@7.5.0 build_type=RelWithDebInfo ^openblas threads=openmp ^mpich"
mkdir build
cd build
spack --color always -e ci build-env $SPEC -- cmake -DCMAKE_INSTALL_PREFIX=$HOME/local -DBUILD_TESTS=1 ..
spack --color always -e ci build-env $SPEC -- make -j 2 install
