/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file diagonalize.hpp
 *
 *  \brief Entry point for Hamiltonain diagonalization.
 */

#ifndef __DIAGONALIZE_HPP__
#define __DIAGONALIZE_HPP__

#include "davidson_result.hpp"

namespace sirius {

template <typename T>
class Hamiltonian0;

class K_point_set;

struct diagonalize_result_t
{
    davidson_result_t davidson_result;
    double avg_num_iter{0};
    bool converged;
};

/// Diagonalize KS Hamiltonian for all k-points in the set.
/** \tparam T  Precision type of the wave-functions
 *  \tparam F  Precition type of the Hamiltonian matrix
 */
/** Defined in diagonalize.cpp and instantiated for <double, double> (and <float, float>, <float, double> with
 *  SIRIUS_USE_FP32). */
template <typename T, typename F>
diagonalize_result_t
diagonalize(Hamiltonian0<T> const& H0__, K_point_set& kset__, double itsol_tol__, int itsol_num_steps__);

/// Initialize the wave-functions of all k-points in the set (atomic orbitals or random).
/** Defined in diagonalize.cpp and instantiated for double (and float with SIRIUS_USE_FP32). */
template <typename T>
void
initialize_subspace(K_point_set& kset__, Hamiltonian0<T>& H0__);

} // namespace sirius

#endif
