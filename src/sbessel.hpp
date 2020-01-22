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

/** \file sbessel.hpp
 *
 *  \brief Contains implementation of sirius::Spherical_Bessel_functions classe.
 */

#ifndef __SBESSEL_HPP__
#define __SBESSEL_HPP__

#include <vector>
#include "spline.hpp"
#include "radial_grid.hpp"

namespace sirius {

/// Spherical Bessel functions \f$ j_{\ell}(q x) \f$ up to lmax.
class Spherical_Bessel_functions
{
  private:
    int lmax_{-1};

    double q_{0};

    Radial_grid<double> const* rgrid_{nullptr};

    std::vector<Spline<double>> sbessel_;

  public:
    Spherical_Bessel_functions() {}

    Spherical_Bessel_functions(int lmax__, Radial_grid<double> const& rgrid__, double q__);

    static void sbessel(int lmax__, double t__, double* jl__);

    static void sbessel_deriv_q(int lmax__, double q__, double x__, double* jl_dq__);

    Spline<double> const& operator[](int l__) const;

    /// Derivative of Bessel function with respect to q.
    /** \f[
        \frac{\partial j_{\ell}(q x)}{\partial q} = \frac{\ell}{q} j_{\ell}(q x) - x j_{\ell+1}(q x)
        \f]
     */
    Spline<double> deriv_q(int l__);

};

}; // namespace sirius

#endif
