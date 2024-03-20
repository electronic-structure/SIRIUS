/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __PYTHON_MODULE_INCLUDES_HPP__
#define __PYTHON_MODULE_INCLUDES_HPP__

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sirius.hpp>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/complex.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <memory>
#include <stdexcept>
// #ifdef _OPENMP
// #include <omp.h>
// #endif

#include "core/json.hpp"
#include "dft/energy.hpp"
#include "beta_projectors/beta_projectors_base.hpp"
#include "nlcglib/inverse_overlap.hpp"
#include "nlcglib/preconditioner/ultrasoft_precond_k.hpp"

#endif /* __PYTHON_MODULE_INCLUDES_HPP__ */
