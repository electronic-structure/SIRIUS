/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file generate_sbessel_mt.hpp
 *
 *  \brief Generate spherical Bessel functions at the muffin-tin boundary for the local set of G-vectors.
 */

#ifndef __GENERATE_SBESSEL_MT_HPP__
#define __GENERATE_SBESSEL_MT_HPP__

namespace sirius {

/// Compute values of spherical Bessel functions at MT boundary.
inline auto
generate_sbessel_mt(Simulation_context const& ctx__, int lmax__)
{
    PROFILE("sirius::generate_sbessel_mt");

    mdarray<double, 3> sbessel_mt({lmax__ + 1, ctx__.gvec().count(), ctx__.unit_cell().num_atom_types()});
    for (int iat = 0; iat < ctx__.unit_cell().num_atom_types(); iat++) {
        #pragma omp parallel for schedule(static)
        for (auto it : ctx__.gvec()) {
            auto gv = ctx__.gvec().gvec_cart(it.igloc);
            gsl_sf_bessel_jl_array(lmax__, gv.length() * ctx__.unit_cell().atom_type(iat).mt_radius(),
                                   &sbessel_mt(0, it.igloc, iat));
        }
    }
    return sbessel_mt;
}

} // namespace sirius

#endif
