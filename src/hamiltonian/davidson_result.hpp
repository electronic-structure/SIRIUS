/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file davidson_result.hpp
 *
 *  \brief Result of the Davidson solver.
 */

#ifndef __DAVIDSON_RESULT_HPP__
#define __DAVIDSON_RESULT_HPP__

#include "core/memory.hpp"

namespace sirius {

/// Result of Davidson solver.
struct davidson_result_t
{
    /// Number of iterations.
    int niter;
    /// Eigen-values.
    mdarray<double, 2> eval;
    /// True if all bands (up and dn) are converged.
    bool converged;
    /// Number of unconverged bands for each spin channel.
    int num_unconverged[2];
};

} // namespace sirius

#endif
