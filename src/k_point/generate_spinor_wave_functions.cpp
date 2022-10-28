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

#include "k_point/k_point.hpp"
#include "wf_trans.hpp"

namespace sirius {

template <typename T>
void K_point<T>::generate_spinor_wave_functions()
{
    PROFILE("sirius::K_point::generate_spinor_wave_functions");

    if (ctx_.cfg().control().use_second_variation()) {
        int nfv = ctx_.num_fv_states();

        if (!ctx_.need_sv()) {
            /* copy eigen-states and exit */
            wf::copy(sddk::memory_t::host, *fv_states_, wf::spin_index(0), wf::band_range(0, ctx_.num_fv_states()),
                     *spinor_wave_functions_ , wf::spin_index(0), wf::band_range(0, ctx_.num_fv_states()));
            return;
        }

        int nbnd = (ctx_.num_mag_dims() == 3) ? ctx_.num_bands() : nfv;

        if (ctx_.processing_unit() == sddk::device_t::GPU) {
            //fv_states().allocate(sddk::spin_range(0), get_memory_pool(sddk::memory_t::device));
            ////fv_states().copy_to(sddk::spin_range(0), sddk::memory_t::device, 0, nfv);
            sv_eigen_vectors_[0].allocate(get_memory_pool(sddk::memory_t::device)).copy_to(sddk::memory_t::device);
            if (ctx_.num_mag_dims() == 1) {
                sv_eigen_vectors_[1].allocate(get_memory_pool(sddk::memory_t::device)).copy_to(sddk::memory_t::device);
            }
            //if (is_device_memory(ctx_.preferred_memory_t())) {
            //    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            //        spinor_wave_functions().allocate(sddk::spin_range(ispn), get_memory_pool(sddk::memory_t::device));
            //        spinor_wave_functions().copy_to(sddk::spin_range(ispn), sddk::memory_t::device, 0, nbnd);
            //    }
            //}
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
            wf::transform(ctx_.spla_context(), sddk::memory_t::host, sv_eigen_vectors_[s], o, 0, 1.0, *fv_states_,
                    wf::spin_index(0), wf::band_range(0, nfv), 0.0, *spinor_wave_functions_, wf::spin_index(ispn),
                    wf::band_range(0, nbnd));
        }

        if (ctx_.processing_unit() == sddk::device_t::GPU) {
            sv_eigen_vectors_[0].deallocate(sddk::memory_t::device);
            if (ctx_.num_mag_dims() == 3) {
                sv_eigen_vectors_[1].deallocate(sddk::memory_t::device);
            }
        }
    } else {
        RTE_THROW("not implemented");
    }
}

template void K_point<double>::generate_spinor_wave_functions();
#ifdef USE_FP32
template void K_point<float>::generate_spinor_wave_functions();
#endif

} // namespace sirius
