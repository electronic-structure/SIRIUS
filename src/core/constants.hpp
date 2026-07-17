/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file constants.hpp
 *
 *  \brief Various constants
 */

#ifndef __CONSTANTS_HPP__
#define __CONSTANTS_HPP__

#include <complex>

namespace sirius {

/// NIST value for the inverse fine structure (http://physics.nist.gov/cuu/Constants/index.html)
constexpr double speed_of_light = 137.035999139;

// This value reproduces NIST ScRLDA total energy much better.
// constexpr double speed_of_light = 137.0359895;

constexpr double rest_energy = speed_of_light * speed_of_light;

constexpr double sq_alpha_half = 0.5 / rest_energy;

/// Bohr radius in angstroms.
constexpr double bohr_radius = 0.52917721067;

/// \f$ \pi \f$
constexpr double pi = 3.1415926535897932385;

/// \f$ 2\pi \f$
constexpr double twopi = 6.2831853071795864769;

/// \f$ 4\pi \f$
constexpr double fourpi = 12.566370614359172954;

/// First spherical harmonic \f$ Y_{00} = \frac{1}{\sqrt{4\pi}} \f$.
constexpr double y00 = 0.28209479177387814347;

/// Hartree in electron-volt units.
constexpr double ha2ev = 27.21138505;

const char* const storage_file_name = "sirius.h5";

/// Pauli matrices in {I, Z, X, Y} order.
std::complex<double> const pauli_matrix[4][2][2] = {
        {{1.0, 0.0}, {0.0, 1.0}},
        {{1.0, 0.0}, {0.0, -1.0}},
        {{0.0, 1.0}, {1.0, 0.0}},
        {{0.0, std::complex<double>(0, -1)}, {std::complex<double>(0, 1), 0.0}}};

} // namespace sirius

#endif // __CONSTANTS_HPP__
