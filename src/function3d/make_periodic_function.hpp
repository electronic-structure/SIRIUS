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

/** \file make_periodic_function.hpp
 *
 *  \brief Generate plane-wave coefficients of the periodic function from the form-factors.
 */

#ifndef __MAKE_PERIODIC_FUNCTION_HPP__
#define __MAKE_PERIODIC_FUNCTION_HPP__

namespace sirius {

/// Make periodic function out of form factors.
/** Return vector of plane-wave coefficients */
template <bool gvec_local, typename F>
inline auto
make_periodic_function(Unit_cell const& uc__, fft::Gvec const& gv__,
                       mdarray<std::complex<double>, 2> const& phase_factors_t__, F&& form_factors__)
{
    PROFILE("sirius::make_periodic_function");

    const double fourpi_omega = fourpi / uc__.omega();

    auto const ngv = gvec_local ? gv__.count() : gv__.num_gvec();
    mdarray<std::complex<double>, 1> f_pw({ngv});
    f_pw.zero();

    #pragma omp parallel for schedule(static)
    for (int igloc = 0; igloc < gv__.count(); igloc++) {
        /* global index of G-vector */
        const int ig   = gv__.offset() + igloc;
        const double g = gv__.gvec_len(gvec_index_t::local(igloc));

        auto const j = gvec_local ? igloc : ig;
        for (int iat = 0; iat < uc__.num_atom_types(); iat++) {
            f_pw[j] += fourpi_omega * std::conj(phase_factors_t__(igloc, iat)) * form_factors__(iat, g);
        }
    }

    if (!gvec_local) {
        gv__.comm().allgather(&f_pw[0], gv__.count(), gv__.offset());
    }

    return f_pw;
}

/// Make periodic out of form factors computed for G-shells.
template <bool gvec_local>
inline auto
make_periodic_function(Unit_cell const& uc__, fft::Gvec const& gv__,
                       mdarray<std::complex<double>, 2> const& phase_factors_t__,
                       mdarray<double, 2> const& form_factors__)
{
    PROFILE("sirius::make_periodic_function");

    const double fourpi_omega = fourpi / uc__.omega();

    auto const ngv = gvec_local ? gv__.count() : gv__.num_gvec();
    mdarray<std::complex<double>, 1> f_pw({ngv});
    f_pw.zero();

    #pragma omp parallel for schedule(static)
    for (int igloc = 0; igloc < gv__.count(); igloc++) {
        /* global index of G-vector */
        const int ig   = gv__.offset() + igloc;
        const int igsh = gv__.shell(ig);

        auto const j = gvec_local ? igloc : ig;
        for (int iat = 0; iat < uc__.num_atom_types(); iat++) {
            f_pw[j] += fourpi_omega * std::conj(phase_factors_t__(igloc, iat)) * form_factors__(igsh, iat);
        }
    }

    if (!gvec_local) {
        gv__.comm().allgather(&f_pw[0], gv__.count(), gv__.offset());
    }

    return f_pw;
}

} // namespace sirius

#endif
