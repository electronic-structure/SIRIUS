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

/** \file check_wave_functions.hpp
 *
 *  \brief Check orthogonality of wave-functions and their residuals.
 */

#ifndef __CHECK_WAVE_FUNCTIONS_HPP__
#define __CHECK_WAVE_FUNCTIONS_HPP__

#include "SDDK/wave_functions.hpp"
#include "hamiltonian/hamiltonian.hpp"

namespace sirius {

template <typename T, typename F>
void check_wave_functions(Hamiltonian_k<real_type<T>>& Hk__, wf::Wave_functions<T>& psi__, wf::spin_range sr__,
        wf::band_range br__, double* eval__)
{
    wf::Wave_functions<T> hpsi(psi__.gkvec_sptr(), psi__.num_md(), wf::num_bands(br__.size()), sddk::memory_t::host);
    wf::Wave_functions<T> spsi(psi__.gkvec_sptr(), psi__.num_md(), wf::num_bands(br__.size()), sddk::memory_t::host);

    la::dmatrix<F> ovlp(br__.size(), br__.size());

    /* apply Hamiltonian and S operators to the wave-functions */
    Hk__.template apply_h_s<F>(sr__, br__, psi__, &hpsi, &spsi);

    wf::inner(Hk__.H0().ctx().spla_context(), sddk::memory_t::host, sr__, psi__, br__, spsi, br__, ovlp, 0, 0);

    auto diff = check_identity(ovlp, br__.size());
    if (diff > 1e-12) {
        std::cout << "[check_wave_functions] Error! Wave-functions are not orthonormal, difference : " << diff << std::endl;
    } else {
        std::cout << "[check_wave_functions] Ok! Wave-functions are orthonormal" << std::endl;
    }

    for (int ib = 0; ib < br__.size(); ib++) {
        double l2norm{0};
        for (auto s = sr__.begin(); s != sr__.end(); s++) {
            auto s1 = hpsi.actual_spin_index(s);
            for (int ig = 0; ig < psi__.gkvec().count(); ig++) {
                /* H|psi> - e S|psi> */
                auto z = hpsi.pw_coeffs(ig, s1, wf::band_index(ib)) -
                    spsi.pw_coeffs(ig, s1, wf::band_index(ib)) * static_cast<real_type<T>>(eval__[ib]);
                l2norm += std::real(z * std::conj(z));
            }
            psi__.gkvec().comm().allreduce(&l2norm, 1);
            l2norm = std::sqrt(l2norm);
            std::cout << "[check_wave_functions] band : " << ib << ", residual l2norm : " << l2norm << std::endl;
        }
    }
}

}

#endif

