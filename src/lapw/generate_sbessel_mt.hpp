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
