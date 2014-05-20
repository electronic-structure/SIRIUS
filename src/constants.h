// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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

/** \file constants.h
 *
 *  \brief Various constants
 */

#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

#include "atomic_conf.h"
#include "atomic_symb.h"

const int major_version = 14;
const int minor_version = 5;

/// NIST value for the inverse fine structure (http://physics.nist.gov/cuu/Constants/index.html)
const double speed_of_light = 137.035999074; 

const double pi = 3.1415926535897932385;

const double twopi = 6.2831853071795864769;

const double fourpi = 12.566370614359172954;

const double y00 = 0.28209479177387814347;

const double ha2ev = 27.21138505;

const double pw_cutoff_default = 20.0; 

const double aw_cutoff_default = 7.0;

const int lmax_apw_default = 8;

const int lmax_rho_default = 7;

const int lmax_pot_default = 7;

const char* const storage_file_name = "sirius.h5";

const int _dim_k_ = 0;

const int _dim_col_ = 1;

const int _dim_row_ = 2;

const std::complex<double> complex_one(1, 0);

const std::complex<double> complex_i(0, 1);

const std::complex<double> complex_zero(0, 0);

#endif // __CONSTANTS_H__

