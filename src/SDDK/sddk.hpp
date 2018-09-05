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

#include "profiler.hpp"
#include "communicator.hpp"
#ifdef __GPU
#include "GPU/cuda.hpp"
#endif
#include "mpi_grid.hpp"
#include "blacs_grid.hpp"
#include "splindex.hpp"
#include "mdarray.hpp"
#include "dmatrix.hpp"
#include "matrix_storage.hpp"
#include "gvec.hpp"
#include "fft3d.hpp"
#include "wave_functions.hpp"

#endif
