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

/** \file generate_pw_coefs.hpp
 *
 *  \brief Generate plane-wave coefficients of the potential for the LAPW Hamiltonian.
 */

inline void Potential::generate_pw_coefs()
{
    PROFILE("sirius::Potential::generate_pw_coefs");

    double sq_alpha_half = 0.5 * std::pow(speed_of_light, -2);

    int gv_count  = ctx_.gvec_partition().gvec_count_fft();

    /* temporaty output buffer */
    mdarray<double_complex, 1> fpw_fft(gv_count);

    switch (ctx_.valence_relativity()) {
        case relativity_t::iora: {
            for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
                double M = 1 - sq_alpha_half * effective_potential().f_rg(ir);
                ctx_.fft().buffer(ir) = ctx_.theta(ir) / std::pow(M, 2);
            }
            if (ctx_.fft().pu() == GPU) {
                ctx_.fft().buffer().copy<memory_t::host, memory_t::device>();
            }
            ctx_.fft().transform<-1>(&fpw_fft[0]);
            ctx_.gvec_partition().gather_pw_global(&fpw_fft[0], &rm2_inv_pw_[0]);
        }
        case relativity_t::zora: {
            for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
                double M = 1 - sq_alpha_half * effective_potential().f_rg(ir);
                ctx_.fft().buffer(ir) = ctx_.theta(ir) / M;
            }
            if (ctx_.fft().pu() == GPU) {
                ctx_.fft().buffer().copy<memory_t::host, memory_t::device>();
            }
            ctx_.fft().transform<-1>(&fpw_fft[0]);
            ctx_.gvec_partition().gather_pw_global(&fpw_fft[0], &rm_inv_pw_[0]);
        }
        default: {
            for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
                ctx_.fft().buffer(ir) = effective_potential().f_rg(ir) * ctx_.theta(ir);
            }
            if (ctx_.fft().pu() == GPU) {
                ctx_.fft().buffer().copy<memory_t::host, memory_t::device>();
            }
            ctx_.fft().transform<-1>(&fpw_fft[0]);
            ctx_.gvec_partition().gather_pw_global(&fpw_fft[0], &veff_pw_[0]);
        }
    }

    /* for full diagonalization we also need Beff(G) */
    if (!use_second_variation) {
        TERMINATE_NOT_IMPLEMENTED
    }
}
