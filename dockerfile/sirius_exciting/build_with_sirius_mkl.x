#!/bin/bash

export SIRIUS_ROOT="$(spack find --format='{prefix}' sirius)"
export SIRIUS_CPP_FLAGS="-I$SIRIUS_ROOT/include/sirius -DSIRIUS"
export MKLROOT="$(spack find --format='{prefix}' intel-oneapi-mkl+cluster)"
export LAPACK_LIB="-L$MKLROOT/mkl/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -Wl,-rpath,$MKLROOT/mkl/lib/intel64"
export SIRIUS_LIB="-L$SIRIUS_ROOT/lib -lsirius -Wl,-rpath,$SIRIUS_ROOT/lib"

cd exciting
spack build-env $SPEC -- make mpiandsmp

