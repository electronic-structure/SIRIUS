/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

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
