/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __WAVEFUNCTION_STRAIN_DERIV_HPP__
#define __WAVEFUNCTION_STRAIN_DERIV_HPP__

#include "context/simulation_context.hpp"

namespace sirius {

void
wavefunctions_strain_deriv(Simulation_context const& ctx__, K_point<double>& kp__, wf::Wave_functions<double>& dphi__,
                           mdarray<double, 2> const& rlm_g__, mdarray<double, 3> const& rlm_dg__, int nu__, int mu__)
{
    auto num_ps_atomic_wf = ctx__.unit_cell().num_ps_atomic_wf();
    PROFILE("sirius::wavefunctions_strain_deriv");
    #pragma omp parallel for schedule(static)
    for (int igkloc = 0; igkloc < kp__.num_gkvec_loc(); igkloc++) {
        /* Cartesian coordinats of G-vector */
        auto gvc = kp__.gkvec().gkvec_cart(gvec_index_t::local(igkloc));
        /* vs = {r, theta, phi} */
        auto gvs = r3::spherical_coordinates(gvc);

        std::vector<mdarray<double, 1>> ri_values(ctx__.unit_cell().num_atom_types());
        for (int iat = 0; iat < ctx__.unit_cell().num_atom_types(); iat++) {
            ri_values[iat] = ctx__.ri().ps_atomic_wf_->values(iat, gvs[0]);
        }

        std::vector<mdarray<double, 1>> ridjl_values(ctx__.unit_cell().num_atom_types());
        for (int iat = 0; iat < ctx__.unit_cell().num_atom_types(); iat++) {
            ridjl_values[iat] = ctx__.ri().ps_atomic_wf_djl_->values(iat, gvs[0]);
        }

        const double p = (mu__ == nu__) ? 0.5 : 0.0;

        for (int ia = 0; ia < ctx__.unit_cell().num_atoms(); ia++) {
            auto& atom_type = ctx__.unit_cell().atom(ia).type();
            // TODO: this can be optimized, check k_point::generate_atomic_wavefunctions()
            auto phase =
                    twopi * dot(kp__.gkvec().gkvec(gvec_index_t::local(igkloc)), ctx__.unit_cell().atom(ia).position());
            auto phase_factor = std::exp(std::complex<double>(0.0, phase));
            for (auto const& e : atom_type.indexb_wfs()) {
                /*  orbital quantum  number of this atomic orbital */
                int l = e.am.l();
                /*  composite l,m index */
                int lm = e.lm;
                /* index of the radial function */
                int idxrf        = e.idxrf;
                int offset_in_wf = num_ps_atomic_wf.second[ia] + e.xi;

                auto z = std::pow(std::complex<double>(0, -1), l) * fourpi / std::sqrt(ctx__.unit_cell().omega());

                /* case |G+k| = 0 */
                if (gvs[0] < 1e-10) {
                    if (l == 0) {
                        auto d1 = ri_values[atom_type.id()][idxrf] * p * y00;

                        dphi__.pw_coeffs(igkloc, wf::spin_index(0), wf::band_index(offset_in_wf)) =
                                -z * d1 * phase_factor;
                    } else {
                        dphi__.pw_coeffs(igkloc, wf::spin_index(0), wf::band_index(offset_in_wf)) = 0.0;
                    }
                } else {
                    auto d1 = ri_values[atom_type.id()][idxrf] *
                              (gvc[mu__] * rlm_dg__(lm, nu__, igkloc) + p * rlm_g__(lm, igkloc));
                    auto d2 =
                            ridjl_values[atom_type.id()][idxrf] * rlm_g__(lm, igkloc) * gvc[mu__] * gvc[nu__] / gvs[0];

                    dphi__.pw_coeffs(igkloc, wf::spin_index(0), wf::band_index(offset_in_wf)) =
                            -z * (d1 + d2) * std::conj(phase_factor);
                }
            } // xi
        }
    }
}

} // namespace sirius

#endif
