#!/bin/bash

mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSpFFT_DIR=/home/travis/local/lib/cmake/SpFFT -DCMAKE_INSTALL_PREFIX=/home/travis/local
make -j 2 install

