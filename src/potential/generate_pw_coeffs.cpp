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

    auto get_fft_result = [&](std::complex<double>* ptr) {
        fft.forward(SPFFT_PU_HOST, reinterpret_cast<double*>(&fpw_fft[0]), SPFFT_FULL_SCALING);
        ctx_.gvec_fft().gather_pw_global(&fpw_fft[0], ptr);
    };

    switch (ctx_.valence_relativity()) {
        case relativity_t::iora: {
            fft::spfft_input<double>(fft, [&](int ir) -> double {
                double M = 1 - sq_alpha_half * effective_potential().rg().value(ir);
                return ctx_.theta(ir) / std::pow(M, 2);
            });
            get_fft_result(&rm2_inv_pw_[0]);
        }
        case relativity_t::zora: {
            fft::spfft_input<double>(fft, [&](int ir) -> double {
                double M = 1 - sq_alpha_half * effective_potential().rg().value(ir);
                return ctx_.theta(ir) / M;
            });
            get_fft_result(&rm_inv_pw_[0]);
        }
        default: {
        }
    }
    if (ctx_.cfg().control().use_second_variation()) {
        fft::spfft_input<double>(
                fft, [&](int ir) -> double { return effective_potential().rg().value(ir) * ctx_.theta(ir); });
        get_fft_result(&veff_pw_[0]);
    } else {
        switch (ctx_.num_mag_dims()) {
            case 3: {
                // spin-block index always has this order
                // 0: V - Bz
                // 1: V + Bz
                // 2: Bx - i By
                // 3: Bx + i By
                fft::spfft_input<std::complex<double>>(fft, [&](int ir) -> std::complex<double> {
                    // Bx - i By
                    return std::complex<double>(effective_magnetic_field(1).rg().value(ir),
                                                -effective_magnetic_field(2).rg().value(ir)) *
                           ctx_.theta(ir);
                });
                get_fft_result(&veff_pw_(0, 2));

                fft::spfft_input<std::complex<double>>(fft, [&](int ir) -> std::complex<double> {
                    // Bx + i By
                    return std::complex<double>(effective_magnetic_field(1).rg().value(ir),
                                                effective_magnetic_field(2).rg().value(ir)) *
                           ctx_.theta(ir);
                });
                get_fft_result(&veff_pw_(0, 3));
            }
            case 1: {
                fft::spfft_input<double>(fft, [&](int ir) -> double {
                    // V + Bz
                    return (effective_potential().rg().value(ir) + effective_magnetic_field(0).rg().value(ir)) *
                           ctx_.theta(ir);
                });
                get_fft_result(&veff_pw_(0, 0));

                fft::spfft_input<double>(fft, [&](int ir) -> double {
                    // V - Bz
                    return (effective_potential().rg().value(ir) - effective_magnetic_field(0).rg().value(ir)) *
                           ctx_.theta(ir);
                });
                get_fft_result(&veff_pw_(0, 1));

                break;
            }
            case 0: {
                fft::spfft_input<double>(
                        fft, [&](int ir) -> double { return effective_potential().rg().value(ir) * ctx_.theta(ir); });
                get_fft_result(&veff_pw_(0, 0));
            }
        }
    }
}

} // namespace sirius
