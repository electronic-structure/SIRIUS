/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file apply_hamiltonian.hpp
 *
 *  \brief Helper function for nlcglib.
 */

#ifndef __APPLY_HAMILTONIAN_HPP__
#define __APPLY_HAMILTONIAN_HPP__

#include "potential/potential.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "density/density.hpp"
#include "core/wf/wave_functions.hpp"
#include <memory>
#include <complex>

namespace sirius {

template <bool gamma_point = false>
inline void
apply_hamiltonian(Hamiltonian0<double>& H0, K_point<double>& kp, wf::Wave_functions<double>& wf_out,
                  wf::Wave_functions<double>& wf, std::shared_ptr<wf::Wave_functions<double>> swf)
{
    int num_wf = wf.num_wf();
    int num_sc = wf.num_sc();
    if (num_wf != wf_out.num_wf() || wf_out.num_sc() != num_sc) {
        RTE_THROW("num_sc or num_wf do not match");
    }
    Hamiltonian_k<double> H = H0(kp);
    auto& ctx               = H0.ctx();

    /* apply H to all wave functions */
    int N = 0;
    int n = num_wf;
    for (int ispn_step = 0; ispn_step < ctx.num_spinors(); ispn_step++) {
        // sping_range: 2 for non-collinear magnetism, otherwise ispn_step
        auto spin_range = wf::spin_range((ctx.num_mag_dims() == 3) ? 2 : ispn_step);
        if constexpr (gamma_point) {
            H.apply_h_s<double>(spin_range, wf::band_range(N, n), wf, &wf_out, swf.get());
        } else {
            H.apply_h_s<std::complex<double>>(spin_range, wf::band_range(N, n), wf, &wf_out, swf.get());
        }
    }
}

} // namespace sirius

#endif /* __APPLY_HAMILTONIAN_HPP__ */
