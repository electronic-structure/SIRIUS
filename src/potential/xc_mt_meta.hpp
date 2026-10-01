/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __XC_MT_META_HPP__
#define __XC_MT_META_HPP__

#include "function3d/spheric_function.hpp"
#include "xc_functional.hpp"

namespace sirius {

struct spheric_xc_derivatives_t
{
    double energy{0};
    std::array<std::vector<Flm>, 2> fields; // Raw derivatives with respect to {rho, tau} radial samples.
};

/// Semilocal XC energy and exact derivatives of its radial-spline/Lebedev quadrature.
/** rho__ and tau__ use canonical spin channels, not total/magnetic components.
 *  The returned fields already include radial and angular integration weights.
 *  Gradients are differentiated through the same radial spline used forwards.
 */
spheric_xc_derivatives_t
xc_mt_meta(SHT const& sht__, std::vector<XC_functional> const& functionals__, std::vector<Flm> const& rho__,
           std::vector<Flm> const& tau__);

} // namespace sirius
#endif
