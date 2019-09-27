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

/** \file generate_spinor_wave_functions.hpp
 *
 *  \brief Generate LAPW spinor wave functions from first-variational states.
 */

#include "K_point/k_point.hpp"
#include "wf_trans.hpp"

namespace sirius {

void K_point::generate_spinor_wave_functions()
{
    PROFILE("sirius::K_point::generate_spinor_wave_functions");

    if (ctx_.control().use_second_variation_) {
        int nfv = ctx_.num_fv_states();

        if (!ctx_.need_sv()) {
            /* copy eigen-states and exit */
            spinor_wave_functions().copy_from(device_t::CPU, ctx_.num_fv_states(), fv_states(), 0, 0, 0, 0);
            return;
        }

        int nbnd = (ctx_.num_mag_dims() == 3) ? ctx_.num_bands() : nfv;

        if (ctx_.processing_unit() == device_t::GPU) {
            fv_states().allocate(spin_range(0), memory_t::device);
            fv_states().copy_to(spin_range(0), memory_t::device, 0, nfv);
            sv_eigen_vectors_[0].allocate(memory_t::device).copy_to(memory_t::device);
            if (ctx_.num_mag_dims() == 1) {
                sv_eigen_vectors_[1].allocate(memory_t::device).copy_to(memory_t::device);
            }
            if (is_device_memory(ctx_.preferred_memory_t())) {
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    spinor_wave_functions().allocate(spin_range(ispn), memory_t::device);
                    spinor_wave_functions().copy_to(spin_range(ispn), memory_t::device, 0, nbnd);
                }
            }
        }

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            int s, o;

            if (ctx_.num_mag_dims() == 3) {
                /* in case of non-collinear magnetism sv_eigen_vectors is a single 2Nx2N matrix */
                s = 0;
                o = ispn * nfv; /* offset for spin up is 0, for spin dn is nfv */
            } else {
                /* sv_eigen_vectors is composed of two NxN matrices */
                s = ispn;
                o = 0;
            }
            /* multiply consecutively up and dn blocks */
            transform(ctx_.preferred_memory_t(), ctx_.blas_linalg_t(), ispn, fv_states(), 0, nfv, sv_eigen_vectors_[s],
                      o, 0, spinor_wave_functions(), 0, nbnd);
        }

        if (ctx_.processing_unit() == device_t::GPU) {
            fv_states().deallocate(spin_range(0), memory_t::device);
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                spinor_wave_functions().copy_to(spin_range(ispn), memory_t::host, 0, nbnd);
            }
            sv_eigen_vectors_[0].deallocate(memory_t::device);
            if (ctx_.num_mag_dims() == 3) {
                sv_eigen_vectors_[1].deallocate(memory_t::device);
            }
        }
    } else {
        throw std::runtime_error("not implemented");
    }
}

} // namespace sirius
