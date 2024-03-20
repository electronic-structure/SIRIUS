/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file sbessel.hpp
 *
 *  \brief Contains implementation of sirius::Spherical_Bessel_functions class.
 */

#ifndef __SBESSEL_HPP__
#define __SBESSEL_HPP__

#include <vector>
#include "radial/spline.hpp"
#include "radial/radial_grid.hpp"

namespace sirius {

namespace sf {

/// Spherical Bessel functions \f$ j_{\ell}(q x) \f$ up to lmax.
class Spherical_Bessel_functions
{
  private:
    double q_{0};

    Radial_grid<double> const* rgrid_{nullptr};

    std::vector<Spline<double>> sbessel_;

  public:
    Spherical_Bessel_functions()
    {
    }

    Spherical_Bessel_functions(int lmax__, Radial_grid<double> const& rgrid__, double q__);

    static void
    sbessel(int lmax__, double t__, double* jl__);

    static void
    sbessel_deriv_q(int lmax__, double q__, double x__, double* jl_dq__);

    Spline<double> const&
    operator[](int l__) const;

    /// Derivative of Bessel function with respect to q.
    /** \f[
        \frac{\partial j_{\ell}(q x)}{\partial q} = \frac{\ell}{q} j_{\ell}(q x) - x j_{\ell+1}(q x)
        \f]
     */
    Spline<double>
    deriv_q(int l__);
};

} // namespace sf

}; // namespace sirius

#endif
