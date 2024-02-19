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

/** \file generate_pw_coefs.cpp
 *
 *  \brief Generate plane-wave coefficients of the potential for the LAPW Hamiltonian.
 */

#include "potential.hpp"

namespace sirius {

void
Potential::generate_pw_coefs()
{
    PROFILE("sirius::Potential::generate_pw_coefs");

    double sq_alpha_half = 0.5 * std::pow(speed_of_light, -2);

    int gv_count = ctx_.gvec_fft().count();

    auto& fft = ctx_.spfft<double>();

    /* temporaty output buffer */
    mdarray<std::complex<double>, 1> fpw_fft({gv_count});

    switch (ctx_.valence_relativity()) {
        case relativity_t::iora: {
            fft::spfft_input<double>(fft, [&](int ir) -> double {
                double M = 1 - sq_alpha_half * effective_potential().rg().value(ir);
                return ctx_.theta(ir) / std::pow(M, 2);
            });
            fft.forward(SPFFT_PU_HOST, reinterpret_cast<double*>(&fpw_fft[0]), SPFFT_FULL_SCALING);
            ctx_.gvec_fft().gather_pw_global(&fpw_fft[0], &rm2_inv_pw_[0]);
        }
        case relativity_t::zora: {
            fft::spfft_input<double>(fft, [&](int ir) -> double {
                double M = 1 - sq_alpha_half * effective_potential().rg().value(ir);
                return ctx_.theta(ir) / M;
            });
            fft.forward(SPFFT_PU_HOST, reinterpret_cast<double*>(&fpw_fft[0]), SPFFT_FULL_SCALING);
            ctx_.gvec_fft().gather_pw_global(&fpw_fft[0], &rm_inv_pw_[0]);
        }
        default: {
            fft::spfft_input<double>(
                    fft, [&](int ir) -> double { return effective_potential().rg().value(ir) * ctx_.theta(ir); });
            fft.forward(SPFFT_PU_HOST, reinterpret_cast<double*>(&fpw_fft[0]), SPFFT_FULL_SCALING);
            ctx_.gvec_fft().gather_pw_global(&fpw_fft[0], &veff_pw_[0]);
        }
    }

    /* for full diagonalization we also need Beff(G) */
    if (!ctx_.cfg().control().use_second_variation()) {
        throw std::runtime_error("not implemented");
    }
}

} // namespace sirius
