// Copyright (c) 2013-2019 Anton Kozhevnikov, Thomas Schulthess
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

/** \file get_h_o_diag.hpp
 *
 *  \brief Get diagonal values of Hamiltonian and overlap matrices.
 */
#include "Unit_cell/unit_cell.hpp"
#include "K_point/k_point.hpp"
#include "Hamiltonian/hamiltonian.hpp"

namespace sirius {

//template<typename T>
//inline mdarray<double, 2>
//Hamiltonian::get_h_diag(K_point *kp__) const
//{
//    PROFILE("sirius::Hamiltonian::get_h_diag");
//
//    mdarray<double, 2> h_diag(kp__->num_gkvec_loc(), ctx_.num_spins());
//
//    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
//
//        /* local H contribution */
//        for (int ig_loc = 0; ig_loc < kp__->num_gkvec_loc(); ig_loc++) {
//            auto vgk = kp__->gkvec().gkvec_cart<index_domain_t::local>(ig_loc);
//            h_diag(ig_loc, ispn) = 0.5 * dot(vgk, vgk) + this->local_op().v0(ispn);
//        }
//
//        /* non-local H contribution */
//        auto beta_gk_t = kp__->beta_projectors().pw_coeffs_t(0);
//        matrix<double_complex> beta_gk_tmp(unit_cell_.max_mt_basis_size(), kp__->num_gkvec_loc());
//
//        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
//            auto &atom_type = unit_cell_.atom_type(iat);
//            int nbf = atom_type.mt_basis_size();
//            matrix<T> d_sum(nbf, nbf);
//            d_sum.zero();
//
//            for (int i = 0; i < atom_type.num_atoms(); i++) {
//                int ia = atom_type.atom_id(i);
//
//                for (int xi2 = 0; xi2 < nbf; xi2++) {
//                    for (int xi1 = 0; xi1 < nbf; xi1++) {
//                        d_sum(xi1, xi2) += this->D().value<T>(xi1, xi2, ispn, ia);
//                    }
//                }
//            }
//
//            int offs = unit_cell_.atom_type(iat).offset_lo();
//            for (int ig_loc = 0; ig_loc < kp__->num_gkvec_loc(); ig_loc++) {
//                for (int xi = 0; xi < nbf; xi++) {
//                    beta_gk_tmp(xi, ig_loc) = beta_gk_t(ig_loc, offs + xi);
//                }
//            }
//
//            #pragma omp parallel for schedule(static)
//            for (int ig_loc = 0; ig_loc < kp__->num_gkvec_loc(); ig_loc++) {
//                for (int xi2 = 0; xi2 < nbf; xi2++) {
//                    for (int xi1 = 0; xi1 < nbf; xi1++) {
//                        /* compute <G+k|beta_xi1> D_{xi1, xi2} <beta_xi2|G+k> contribution from all atoms */
//                        auto z = beta_gk_tmp(xi1, ig_loc) * d_sum(xi1, xi2) * std::conj(beta_gk_tmp(xi2, ig_loc));
//                        h_diag(ig_loc, ispn) += z.real();
//                    }
//                }
//            }
//        }
//    }
//    if (ctx_.processing_unit() == device_t::GPU) {
//        h_diag.allocate(memory_t::device).copy_to(memory_t::device);
//    }
//    return h_diag;
//}
//
//template<typename T>
//mdarray<double, 1>
//Hamiltonian::get_o_diag(K_point *kp__) const // TODO: this is not strictly true for SO case, but it is used only in
//                                             //       preconditioning, so it's ok
//{
//    PROFILE("sirius::Hamiltonian::get_o_diag");
//
//    mdarray<double, 1> o_diag(kp__->num_gkvec_loc());
//    for (int ig = 0; ig < kp__->num_gkvec_loc(); ig++) {
//        o_diag[ig] = 1;
//    }
//
//    /* non-local O contribution */
//    auto beta_gk_t = kp__->beta_projectors().pw_coeffs_t(0);
//    matrix<double_complex> beta_gk_tmp(unit_cell_.max_mt_basis_size(), kp__->num_gkvec_loc());
//
//    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
//        auto &atom_type = unit_cell_.atom_type(iat);
//        if (!atom_type.augment()) {
//            continue;
//        }
//
//        int nbf = atom_type.mt_basis_size();
//
//        matrix<T> q_sum(nbf, nbf);
//        q_sum.zero();
//
//        for (int i = 0; i < atom_type.num_atoms(); i++) {
//            int ia = atom_type.atom_id(i);
//
//            for (int xi2 = 0; xi2 < nbf; xi2++) {
//                for (int xi1 = 0; xi1 < nbf; xi1++) {
//                    q_sum(xi1, xi2) += Q().value<T>(xi1, xi2, ia);
//                }
//            }
//        }
//
//        int offs = unit_cell_.atom_type(iat).offset_lo();
//        for (int ig_loc = 0; ig_loc < kp__->num_gkvec_loc(); ig_loc++) {
//            for (int xi = 0; xi < nbf; xi++) {
//                beta_gk_tmp(xi, ig_loc) = beta_gk_t(ig_loc, offs + xi);
//            }
//        }
//
//        #pragma omp parallel for
//        for (int ig_loc = 0; ig_loc < kp__->num_gkvec_loc(); ig_loc++) {
//            for (int xi2 = 0; xi2 < nbf; xi2++) {
//                for (int xi1 = 0; xi1 < nbf; xi1++) {
//                    /* compute <G+k|beta_xi1> Q_{xi1, xi2} <beta_xi2|G+k> contribution from all atoms */
//                    auto z = beta_gk_tmp(xi1, ig_loc) * q_sum(xi1, xi2) * std::conj(beta_gk_tmp(xi2, ig_loc));
//                    o_diag[ig_loc] += z.real();
//                }
//            }
//        }
//    }
//    if (ctx_.processing_unit() == device_t::GPU) {
//        o_diag.allocate(memory_t::device).copy_to(memory_t::device);
//    }
//    return o_diag;
//}
//
//template
//mdarray<double, 1>
//Hamiltonian::get_o_diag<double>(K_point *kp__) const;
//
//template
//mdarray<double, 1>
//Hamiltonian::get_o_diag<double_complex>(K_point *kp__) const;
//template
//mdarray<double, 2>
//Hamiltonian::get_h_diag<double>(K_point *kp__) const;
//
//template
//mdarray<double, 2>
//Hamiltonian::get_h_diag<double_complex>(K_point *kp__) const;

}
