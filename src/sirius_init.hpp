/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file sirius_init.hpp
 *
 *  \brief Initialization and finalization of the library.
 *
 *  Lightweight alternative to the "all-in-one" sirius.hpp for code which only needs to set up the library.
 */

#ifndef __SIRIUS_INIT_HPP__
#define __SIRIUS_INIT_HPP__

namespace sirius {

/// Return the status of the library (initialized or not).
bool&
is_initialized();

/// Initialize the library.
void
initialize(bool call_mpi_init__ = true);

/// Shut down the library.
void
finalize(bool call_mpi_fin__ = true, bool reset_device__ = true);

} // namespace sirius

#endif // __SIRIUS_INIT_HPP__
