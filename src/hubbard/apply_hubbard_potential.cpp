// Copyright (c) 2013-2020 Mathieu Taillefumier, Anton Kozhevnikov, Thomas Schulthess
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

/** \file apply_hubbard_potential.hpp
 *
 *  \brief Contains implementation of sirius::Hubbard::apply_hubbard_potential() function.
 */

#include "hubbard.hpp"

namespace sirius {

//namespace hubbard {
//void
//apply_potential(spin_range spins__, Hubbard_matrix const& um__, Wave_functions<double> const& hub_wf__, int idx__,
//    int n__, Wave_functions<double> const& phi__, Wave_functions<double>& hphi__)
//{
//    dmatrix<double_complex> dm(hub_wf.num_wf(), n__);
//    dm.zero();
//
//    if (ctx_.processing_unit() == device_t::GPU) {
//        dm.allocate(memory_t::device);
//    }
//
//    /* First calculate the local part of the projections
//       dm(i, n) = <phi_i| S |psi_{nk}> */
//    inner(ctx_.spla_context(), spins__, hub_wf, 0, hub_wf.num_wf(), phi, idx__, n__, dm, 0, 0);
//
//    dmatrix<double_complex> Up(hub_wf.num_wf(), n__);
//    Up.zero();
//
//    #pragma omp parallel for schedule(static)
//    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ++ia) {
//        const auto& atom = ctx_.unit_cell().atom(ia);
//        if (atom.type().hubbard_correction()) {
//            const int lmax_at = 2 * atom.type().indexr_hub().am(0).l() + 1;
//            // we apply the hubbard correction. For now I have no papers
//            // giving me the formula for the SO case so I rely on QE for it
//            // but I do not like it at all
//            if (ctx_.num_mag_dims() == 3) {
//                for (int s1 = 0; s1 < ctx_.num_spins(); s1++) {
//                    for (int s2 = 0; s2 < ctx_.num_spins(); s2++) {
//                        const int ind = (s1 == s2) * s1 + (1 + 2 * s2 + s1) * (s1 != s2);
//
//                        // TODO: replace this with matrix matrix multiplication
//
//                        for (int nbd = 0; nbd < n__; nbd++) {
//                            for (int m1 = 0; m1 < lmax_at; m1++) {
//                                for (int m2 = 0; m2 < lmax_at; m2++) {
//                                    Up(this->number_of_hubbard_orbitals_ * s1 + this->offset_[ia] + m1, nbd) +=
//                                        this->hubbard_potential_(m2, m1, ind, ia) *
//                                        dm(this->number_of_hubbard_orbitals_ * s2 + this->offset_[ia] + m2, nbd);
//                                }
//                            }
//                        }
//                    }
//                }
//            } else {
//                // Conventional LDA or colinear magnetism
//                for (int nbd = 0; nbd < n__; nbd++) {
//                    for (int m1 = 0; m1 < lmax_at; m1++) {
//                        for (int m2 = 0; m2 < lmax_at; m2++) {
//                            Up(this->offset_[ia] + m1, nbd) += this->hubbard_potential_(m2, m1, spins__(), ia) *
//                                dm(this->offset_[ia] + m2, nbd);
//                        }
//                    }
//                }
//            }
//        }
//    }
//
//    if (ctx_.processing_unit() == device_t::GPU) {
//        Up.allocate(memory_t::device);
//        Up.copy_to(memory_t::device);
//    }
//
//    transform<double_complex>(ctx_.spla_context(), spins__(), 1.0, {&hub_wf}, 0, hub_wf.num_wf(),
//                              Up, 0, 0, 1.0, {&hphi}, idx__, n__);
//}
//}

/** This function computes the Hubbard contribution to the hamiltonian and add it to ophi.
 *  The S matrix is already applied to phi_i */
void Hubbard::apply_hubbard_potential(Hubbard_matrix const& um__, Wave_functions<double>& hub_wf, spin_range spins__, const int idx__, const int n__,
                                      Wave_functions<double>& phi, Wave_functions<double>& hphi)
{
    dmatrix<double_complex> dm(hub_wf.num_wf(), n__);
    dm.zero();

    if (ctx_.processing_unit() == device_t::GPU) {
        dm.allocate(memory_t::device);
    }

    /* First calculate the local part of the projections
       dm(i, n) = <phi_i| S |psi_{nk}> */
    inner(ctx_.spla_context(), spins__, hub_wf, 0, hub_wf.num_wf(), phi, idx__, n__, dm, 0, 0);

    dmatrix<double_complex> Up(hub_wf.num_wf(), n__);
    Up.zero();

    #pragma omp parallel for schedule(static)
    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ++ia) {
        const auto& atom = ctx_.unit_cell().atom(ia);
        if (atom.type().hubbard_correction()) {
            const int lmax_at = 2 * atom.type().indexr_hub().am(0).l() + 1;
            // we apply the hubbard correction. For now I have no papers
            // giving me the formula for the SO case so I rely on QE for it
            // but I do not like it at all
            if (ctx_.num_mag_dims() == 3) {
                for (int s1 = 0; s1 < ctx_.num_spins(); s1++) {
                    for (int s2 = 0; s2 < ctx_.num_spins(); s2++) {
                        const int ind = (s1 == s2) * s1 + (1 + 2 * s2 + s1) * (s1 != s2);

                        // TODO: replace this with matrix matrix multiplication

                        for (int nbd = 0; nbd < n__; nbd++) {
                            for (int m1 = 0; m1 < lmax_at; m1++) {
                                for (int m2 = 0; m2 < lmax_at; m2++) {
                                    Up(this->number_of_hubbard_orbitals_ * s1 + this->offset_[ia] + m1, nbd) +=
                                        um__.local(ia)(m2, m1, ind) *
                                        dm(this->number_of_hubbard_orbitals_ * s2 + this->offset_[ia] + m2, nbd);
                                }
                            }
                        }
                    }
                }
            } else {
                // Conventional LDA or colinear magnetism
                for (int nbd = 0; nbd < n__; nbd++) {
                    for (int m1 = 0; m1 < lmax_at; m1++) {
                        for (int m2 = 0; m2 < lmax_at; m2++) {
                            Up(this->offset_[ia] + m1, nbd) += um__.local(ia)(m2, m1, spins__()) *
                                dm(this->offset_[ia] + m2, nbd);
                        }
                    }
                }
            }
        }
    }

    if (ctx_.processing_unit() == device_t::GPU) {
        Up.allocate(memory_t::device);
        Up.copy_to(memory_t::device);
    }

    transform<double_complex>(ctx_.spla_context(), spins__(), 1.0, {&hub_wf}, 0, hub_wf.num_wf(),
                              Up, 0, 0, 1.0, {&hphi}, idx__, n__);
}

}
