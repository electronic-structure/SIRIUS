/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file mixer_functions.hpp
 *
 *  \brief Contains declarations of functions required for mixing.
 */

#ifndef __MIXER_FUNCTIONS_HPP__
#define __MIXER_FUNCTIONS_HPP__

#include "function3d/periodic_function.hpp"
#include "core/memory.hpp"
#include "mixer/mixer.hpp"
#include "hubbard/hubbard_matrix.hpp"
#include "density/density_matrix.hpp"
#include "density/density.hpp"

namespace sirius {

namespace mixer {

FunctionProperties<Periodic_function<double>>
periodic_function_property();

FunctionProperties<Periodic_function<double>>
periodic_function_property_modified(bool use_coarse_gvec__);

FunctionProperties<density_matrix_t>
density_function_property();

FunctionProperties<PAW_density<double>>
paw_density_function_property();

FunctionProperties<Hubbard_matrix>
hubbard_matrix_function_property();

} // namespace mixer

} // namespace sirius

#endif
