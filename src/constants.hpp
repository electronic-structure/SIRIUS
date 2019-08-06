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

/** \file constants.hpp
 *
 *  \brief Various constants
 */

#ifndef __CONSTANTS_HPP__
#define __CONSTANTS_HPP__

#include <complex>

/// NIST value for the inverse fine structure (http://physics.nist.gov/cuu/Constants/index.html)
const double speed_of_light = 137.035999139;

// This value reproduces NIST ScRLDA total energy much better.
// const double speed_of_light = 137.0359895;

/// Bohr radius in angstroms.
const double bohr_radius = 0.52917721067;

/// \f$ \pi \f$
const double pi = 3.1415926535897932385;

/// \f$ 2\pi \f$
const double twopi = 6.2831853071795864769;

/// \f$ 4\pi \f$
const double fourpi = 12.566370614359172954;

/// First spherical harmonic \f$ Y_{00} = \frac{1}{\sqrt{4\pi}} \f$.
const double y00 = 0.28209479177387814347;

/// Hartree in electron-volt units.
const double ha2ev = 27.21138505;

const char* const storage_file_name = "sirius.h5";

/// Pauli matrices in {I, Z, X, Y} order.
const std::complex<double> pauli_matrix[4][2][2] = {
    {{1.0, 0.0}, {0.0, 1.0}},
    {{1.0, 0.0}, {0.0, -1.0}},
    {{0.0, 1.0}, {1.0, 0.0}},
    {{0.0, std::complex<double>(0, -1)}, {std::complex<double>(0, 1), 0.0}}};

#endif // __CONSTANTS_HPP__
