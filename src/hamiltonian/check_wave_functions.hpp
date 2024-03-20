/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file check_wave_functions.hpp
 *
 *  \brief Check orthogonality of wave-functions and their residuals.
 */

#ifndef __CHECK_WAVE_FUNCTIONS_HPP__
#define __CHECK_WAVE_FUNCTIONS_HPP__

#include "core/wf/wave_functions.hpp"
#include "hamiltonian/hamiltonian.hpp"

namespace sirius {

template <typename T, typename F>
void
check_wave_functions(Hamiltonian_k<real_type<T>> const& Hk__, wf::Wave_functions<T>& psi__, wf::spin_range sr__,
                     wf::band_range br__, double* eval__)
{
    wf::Wave_functions<T> hpsi(psi__.gkvec_sptr(), psi__.num_md(), wf::num_bands(br__.size()), memory_t::host);
    wf::Wave_functions<T> spsi(psi__.gkvec_sptr(), psi__.num_md(), wf::num_bands(br__.size()), memory_t::host);

    la::dmatrix<F> ovlp(br__.size(), br__.size());

    /* apply Hamiltonian and S operators to the wave-functions */
    Hk__.template apply_h_s<F>(sr__, br__, psi__, &hpsi, &spsi);

    wf::inner(Hk__.H0().ctx().spla_context(), memory_t::host, sr__, psi__, br__, spsi, br__, ovlp, 0, 0);

    auto diff = check_identity(ovlp, br__.size());
    if (diff > 1e-12) {
        std::cout << "[check_wave_functions] Error! Wave-functions are not orthonormal, difference : " << diff
                  << std::endl;
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

} // namespace sirius

#endif
