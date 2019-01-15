// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file sddk.hpp
 *
 *  \brief Main include file for SDDK library.
 */

#ifndef __SDDK_HPP__
#define __SDDK_HPP__

#include <omp.h>
#include <string>
#include <sstream>
#include <chrono>
#include <map>
#include <vector>
#include <memory>
#include <complex>
#include <algorithm>

#include "../utils/utils.hpp"

#define TERMINATE_NO_GPU          TERMINATE("not compiled with GPU support");
#define TERMINATE_NO_SCALAPACK    TERMINATE("not compiled with ScaLAPACK support");
#define TERMINATE_NOT_IMPLEMENTED TERMINATE("feature is not implemented");

using double_complex = std::complex<double>;

/// SDDK: Slab Data Distribution Kit - a collection of classes and functions to work with wave-functions distributed in slabs.
namespace sddk {

}

#include "profiler.hpp"
#include "communicator.hpp"
#include "mpi_grid.hpp"
#include "blacs_grid.hpp"
#include "splindex.hpp"
#include "memory.hpp"
#include "dmatrix.hpp"
#include "matrix_storage.hpp"
#include "gvec.hpp"
#include "fft3d.hpp"
#include "wave_functions.hpp"

#endif
