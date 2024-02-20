// Copyright (c) 2013-2023 Anton Kozhevnikov, Thomas Schulthess
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
    #pragma omp parallel for schedule(static)
    for (auto it : ctx__.gvec()) {
        auto rtp = r3::spherical_coordinates(ctx__.gvec().gvec_cart(it.igloc));
        sf::spherical_harmonics(lmax__, rtp[1], rtp[2], &gvec_ylm(0, it.igloc));
    }
    return gvec_ylm;
}

} // namespace sirius

#endif
