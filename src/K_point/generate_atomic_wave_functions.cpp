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

/** \file generate_atomic_wave_functions.hpp
 *
 *  \brief Generation of initial guess atomic wave-functions and hubbard orbitals.
 */

#include "K_point/k_point.hpp"

namespace sirius {

void K_point::generate_atomic_wave_functions(const basis_functions_index& index, const int atom, const int offset,
                                             const bool hubbard, Wave_functions& phi)
{
    if (index.size() == 0) {
        return;
    }

    int lmax{0};
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);
        lmax            = std::max(lmax, atom_type.lmax_ps_atomic_wf());
    }
    lmax = std::max(lmax, unit_cell_.lmax());

    auto& atom_type = unit_cell_.atom(atom).type();
    #pragma omp parallel for schedule(static)
    for (int igk_loc = 0; igk_loc < this->num_gkvec_loc(); igk_loc++) {
        /* global index of G+k vector */
        int igk = this->idxgk(igk_loc);
        /* vs = {r, theta, phi} */
        auto vs = SHT::spherical_coordinates(this->gkvec().gkvec_cart<index_domain_t::local>(igk_loc));

        /* compute real spherical harmonics for G+k vector */
        std::vector<double> rlm(utils::lmmax(lmax));
        sf::spherical_harmonics(lmax, vs[1], vs[2], &rlm[0]);

        /* need to find the index of the atom type get values of radial
           integrals for a given G+k vector length */
        sddk::mdarray<double, 1> ri_values;
        ri_values = ctx_.atomic_wf_ri().values(atom_type.id(), vs[0]);

        int n{0};
        for (int xi = 0; xi < index.size();) {
            int l = index[xi].l;
            /* index of the radial function */
            int index_radial_function = index[xi].idxrf;

            const auto phase        = twopi * dot(gkvec().gkvec(igk), unit_cell_.atom(atom).position());
            const auto phase_factor = std::exp(double_complex(0.0, phase));
            const auto& atom_type   = unit_cell_.atom(atom).type();
            const auto z            = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());

            if (!hubbard) {
                /* for spin orbit coupling the initial wave functions of a given
                 * shell are full spinors with a different radial part. the
                 * orbitals j = l - 1/2 and j = l + 1/2 are next to each other
                 * in the index structure, so I explicitely compute the
                 * average */

                for (int m = -l; m <= l; m++) {
                    const int lm = utils::lm(l, m);
                    phi.pw_coeffs(0).prime(igk_loc, n + offset) =
                        z * std::conj(phase_factor) * rlm[lm] * ri_values[index_radial_function];
                    n++;
                }
                xi += 2 * l + 1;
            } else {
                /* hubbard atomic orbitals */

                if (atom_type.spin_orbit_coupling()) {
                    /* in that case each atomic orbital has a distinct j and are
                       considered as independent orbitals. */
                    const auto average = (ri_values[atom_type.hubbard_orbital(index_radial_function).rindex()] +
                                          ri_values[atom_type.hubbard_orbital(index_radial_function + 1).rindex()]);

                    for (int m = -l; m <= l; m++) {
                        const int lm = utils::lm(l, m);
                        phi.pw_coeffs(1).prime(igk_loc, n + offset + 2 * l + 1) =
                            0.5 * z * std::conj(phase_factor) * rlm[lm] * average;
                        phi.pw_coeffs(0).prime(igk_loc, n + offset) =
                            0.5 * z * std::conj(phase_factor) * rlm[lm] * average;
                        n += 1;
                    }
                    xi += 2 * (2 * l + 1);
                    n += 2 * l + 1;
                } else {

                    /* it is a one orbital but with degeneracy 2. we should
                       consider the magnetic and non magnetic case */
                    for (int m = -l; m <= l; m++) {
                        const int lm = utils::lm(l, m);
                        phi.pw_coeffs(0).prime(igk_loc, n + offset) =
                            z * std::conj(phase_factor) * rlm[lm] *
                            ri_values[atom_type.hubbard_orbital(index_radial_function).rindex()];
                        /* copy from spin up */
                        if (ctx_.num_mag_dims() == 3) {
                            phi.pw_coeffs(1).prime(igk_loc, n + offset + 2 * l + 1) =
                                 phi.pw_coeffs(0).prime(igk_loc, n + offset);
                        }
                        n++;
                    }
                    xi += 2 * l + 1;
                }
            }
        }
    }
}

void K_point::compute_gradient_wave_functions(Wave_functions& phi, const int starting_position_i, const int num_wf,
                                              Wave_functions& dphi, const int starting_position_j, const int direction)
{
    std::vector<double_complex> qalpha(this->num_gkvec_loc());

    for (int igk_loc = 0; igk_loc < this->num_gkvec_loc(); igk_loc++) {
        auto G = this->gkvec().gkvec_cart<index_domain_t::local>(igk_loc);

        qalpha[igk_loc] = double_complex(0.0, -G[direction]);
    }

    #pragma omp parallel for schedule(static)
    for (int nphi = 0; nphi < num_wf; nphi++) {
        for (int ispn = 0; ispn < phi.num_sc(); ispn++) {
            for (int igk_loc = 0; igk_loc < this->num_gkvec_loc(); igk_loc++) {
                dphi.pw_coeffs(ispn).prime(igk_loc, nphi + starting_position_j) =
                    qalpha[igk_loc] * phi.pw_coeffs(ispn).prime(igk_loc, nphi + starting_position_i);
            }
        }
    }
}

} // namespace sirius
