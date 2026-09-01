/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file generate_gvec_ylm.hpp
 *
 *  \brief Generate complex spherical harmonics for the local set of G-vectors.
 */

#ifndef __GENERATE_GVEC_YLM_HPP__
#define __GENERATE_GVEC_YLM_HPP__

namespace sirius {

/// Generate complex spherical harmonics for the local set of G-vectors.
inline auto
generate_gvec_ylm(Simulation_context const& ctx__, int lmax__)
{
    PROFILE("sirius::generate_gvec_ylm");

    mdarray<std::complex<double>, 2> gvec_ylm({sf::lmmax(lmax__), ctx__.gvec().count()}, mdarray_label("gvec_ylm"));
    int ngv = ctx__.gvec().count();
    #pragma omp parallel for schedule(static)
    for (int igloc = 0; igloc < ngv; igloc++) {
        auto rtp = r3::spherical_coordinates(ctx__.gvec().gvec_cart(gvec_index_t::local(igloc)));
        sf::spherical_harmonics(lmax__, rtp[1], rtp[2], &gvec_ylm(0, igloc));
    }
    return gvec_ylm;
}

} // namespace sirius

#endif
