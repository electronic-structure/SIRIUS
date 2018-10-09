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

#include <gsl/gsl_sf_bessel.h>
#include "Unit_cell/unit_cell.hpp"

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
    Spherical_Bessel_functions()
    {
    }

    Spherical_Bessel_functions(int lmax__, Radial_grid<double> const& rgrid__, double q__)
        : lmax_(lmax__)
        , q_(q__)
        , rgrid_(&rgrid__)
    {
        assert(q_ >= 0);

        sbessel_ = std::vector<Spline<double>>(lmax__ + 2);
        for (int l = 0; l <= lmax__ + 1; l++) {
            sbessel_[l] = Spline<double>(rgrid__);
        }

        std::vector<double> jl(lmax__ + 2);
        for (int ir = 0; ir < rgrid__.num_points(); ir++) {
            double t = rgrid__[ir] * q__;
            gsl_sf_bessel_jl_array(lmax__ + 1, t, &jl[0]);
            for (int l = 0; l <= lmax__ + 1; l++) {
                sbessel_[l](ir) = jl[l];
            }
        }

        for (int l = 0; l <= lmax__ + 1; l++) {
            sbessel_[l].interpolate();
        }
    }

    static void sbessel(int lmax__, double t__, double* jl__)
    {
        gsl_sf_bessel_jl_array(lmax__, t__, jl__);
    }

    static void sbessel_deriv_q(int lmax__, double q__, double x__, double* jl_dq__)
    {
        std::vector<double> jl(lmax__ + 2);
        sbessel(lmax__ + 1, x__ * q__, &jl[0]);

        for (int l = 0; l <= lmax__; l++) {
            if (q__ != 0) {
                jl_dq__[l] = (l / q__) * jl[l] - x__ * jl[l + 1];
            } else {
                if (l == 1) {
                    jl_dq__[l] = x__ / 3;
                } else {
                    jl_dq__[l] = 0;
                }
            }
        }
    }

    Spline<double> const& operator[](int l__) const
    {
        assert(l__ <= lmax_);
        return sbessel_[l__];
    }

    /// Derivative of Bessel function with respect to q.
    /** \f[
     *    \frac{\partial j_{\ell}(q x)}{\partial q} = \frac{\ell}{q} j_{\ell}(q x) - x j_{\ell+1}(q x)
     *  \f]
     */
    Spline<double> deriv_q(int l__)
    {
        assert(l__ <= lmax_);
        assert(q_ >= 0);
        Spline<double> s(*rgrid_);
        if (q_ != 0) {
            for (int ir = 0; ir < rgrid_->num_points(); ir++) {
                s(ir) = (l__ / q_) * sbessel_[l__](ir) - (*rgrid_)[ir] * sbessel_[l__ + 1](ir);
            }
        } else {
            if (l__ == 1) {
                for (int ir = 0; ir < rgrid_->num_points(); ir++) {
                    s(ir) = (*rgrid_)[ir] / 3;
                }
            }
        }
        s.interpolate();
        return std::move(s);
    }
};

}; // namespace sirius

#endif
