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

/** \file smearing.hpp
 *
 *  \brief Smearing functions used in finding the band occupancies.
 */

#ifndef __SMEARING_HPP__
#define __SMEARING_HPP__

#include <cmath>

namespace smearing {

inline double fermi_dirac(double e)
{
    double kT = 0.001;
    if (e > 100 * kT) {
        return 0.0;
    }
    if (e < -100 * kT) {
        return 1.0;
    }
    return (1.0 / (std::exp(e / kT) + 1.0));
}

inline double gaussian(double e, double delta)
{
    return 0.5 * (1 - std::erf(e / delta));
}

inline double cold(double e)
{
    const double pi = 3.1415926535897932385;
    
    double a = -0.5634;

    if (e < -10.0) {
        return 1.0;
    }
    if (e > 10.0) {
        return 0.0;
    }

    return 0.5 * (1 - std::erf(e)) - 1 - 0.25 * std::exp(-e * e) * (a + 2 * e - 2 * a * e * e) / std::sqrt(pi);
}

}

#endif
