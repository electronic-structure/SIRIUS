/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file generate_spinor_wave_functions.hpp
 *
 *  \brief Generate LAPW spinor wave functions from first-variational states.
 */

#include "k_point/k_point.hpp"

namespace sirius {

template <typename T>
void
K_point<T>::generate_spinor_wave_functions()
{
    PROFILE("sirius::K_point::generate_spinor_wave_functions");

    if (ctx_.cfg().control().use_second_variation()) {
        int nfv = ctx_.num_fv_states();

        if (!ctx_.need_sv()) {
            /* copy eigen-states and exit */
            wf::copy(memory_t::host, *fv_states_, wf::spin_index(0), wf::band_range(0, ctx_.num_fv_states()),
                     *spinor_wave_functions_, wf::spin_index(0), wf::band_range(0, ctx_.num_fv_states()));
            return;
        }

        int nbnd = (ctx_.num_mag_dims() == 3) ? ctx_.num_bands() : nfv;

        if (ctx_.processing_unit() == device_t::GPU) {
            // fv_states().allocate(sddk::spin_range(0), get_memory_pool(memory_t::device));
            ////fv_states().copy_to(sddk::spin_range(0), memory_t::device, 0, nfv);
            sv_eigen_vectors_[0].allocate(get_memory_pool(memory_t::device)).copy_to(memory_t::device);
            if (ctx_.num_mag_dims() == 1) {
                sv_eigen_vectors_[1].allocate(get_memory_pool(memory_t::device)).copy_to(memory_t::device);
            }
            // if (is_device_memory(ctx_.preferred_memory_t())) {
            //     for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            //         spinor_wave_functions().allocate(sddk::spin_range(ispn), get_memory_pool(memory_t::device));
            //         spinor_wave_functions().copy_to(sddk::spin_range(ispn), memory_t::device, 0, nbnd);
            //     }
            // }
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
            wf::transform(ctx_.spla_context(), memory_t::host, sv_eigen_vectors_[s], o, 0, 1.0, *fv_states_,
                          wf::spin_index(0), wf::band_range(0, nfv), 0.0, *spinor_wave_functions_, wf::spin_index(ispn),
                          wf::band_range(0, nbnd));
        }

        if (ctx_.processing_unit() == device_t::GPU) {
            sv_eigen_vectors_[0].deallocate(memory_t::device);
            if (ctx_.num_mag_dims() == 3) {
                sv_eigen_vectors_[1].deallocate(memory_t::device);
            }
        }
    } else {
        RTE_THROW("not implemented");
    }
}

template void
K_point<double>::generate_spinor_wave_functions();
#ifdef SIRIUS_USE_FP32
template void
K_point<float>::generate_spinor_wave_functions();
#endif

} // namespace sirius
