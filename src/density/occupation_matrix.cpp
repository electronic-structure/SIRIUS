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

/** \file occupation_matrix.cpp
 *
 *  \brief Occupation matrix of the LDA+U method.
 */

#include "occupation_matrix.hpp"
#include "symmetry/crystal_symmetry.hpp"

namespace sirius {

Occupation_matrix::Occupation_matrix(Simulation_context& ctx__)
    : Hubbard_matrix(ctx__)
{
    if (!ctx_.hubbard_correction()) {
        return;
    }
    /* a pair of "total number, offests" for the Hubbard orbitals idexing */
    int nhwf = ctx_.unit_cell().num_hubbard_wf().first;

    /* find all possible translations */
    for (int i = 0; i < ctx_.cfg().hubbard().nonlocal().size(); i++) {
        auto nl = ctx_.cfg().hubbard().nonlocal(i);
        int ia  = nl.atom_pair()[0];
        int ja  = nl.atom_pair()[1];
        auto T  = nl.T();

        auto& sym = ctx_.unit_cell().symmetry();

        for (int isym = 0; isym < sym.size(); isym++) {
            auto Ttot = sym[isym].spg_op.inv_sym_atom_T[ja] - sym[isym].spg_op.inv_sym_atom_T[ia] +
                        dot(sym[isym].spg_op.invR, vector3d<int>(T));
            if (!occ_mtrx_T_.count(Ttot)) {
                occ_mtrx_T_[Ttot] = sddk::mdarray<double_complex, 3>(nhwf, nhwf, ctx_.num_mag_comp());
                occ_mtrx_T_[Ttot].zero();
            }
        }
    }
}

template <typename T>
void
Occupation_matrix::add_k_point_contribution(K_point<T>& kp__)
{
    if (!ctx_.hubbard_correction()) {
        return;
    }

    memory_t mem_host{memory_t::host};
    memory_t mem{memory_t::host};
    linalg_t la{linalg_t::blas};
    /* find the appropriate linear algebra provider */
    if (ctx_.processing_unit() == device_t::GPU) {
        mem      = memory_t::device;
        mem_host = memory_t::host_pinned;
        if (is_device_memory(ctx_.preferred_memory_t())) {
            la = linalg_t::gpublas;
        } else {
            la = linalg_t::spla;
        }
    }

    int nwfu = kp__.hubbard_wave_functions_S().num_wf();

    sddk::matrix<std::complex<T>> occ_mtrx(nwfu, nwfu, ctx_.mem_pool(memory_t::host), "occ_mtrx");
    if (is_device_memory(mem)) {
        occ_mtrx.allocate(ctx_.mem_pool(mem));
    }

    /* a pair of "total number, offests" for the Hubbard orbitals idexing */
    auto r = ctx_.unit_cell().num_hubbard_wf();

    // TODO collnear and non-collinear cases have a lot of similar code; there should be a way to combine it

    /* full non collinear magnetism */
    if (ctx_.num_mag_dims() == 3) {
        dmatrix<std::complex<T>> dm(kp__.num_occupied_bands(), nwfu, ctx_.mem_pool(mem_host), "dm");
        if (is_device_memory(mem)) {
            dm.allocate(ctx_.mem_pool(mem));
        }
        sddk::inner(ctx_.spla_context(), spin_range(2), kp__.spinor_wave_functions(), 0, kp__.num_occupied_bands(),
                    kp__.hubbard_wave_functions_S(), 0, nwfu, dm, 0, 0);

        // TODO: check if inner() already moved data to CPU

        // if (is_device_memory(mem)) {
        //    dm.copy_to(memory_t::host);
        //}
        dmatrix<std::complex<T>> dm1(kp__.num_occupied_bands(), nwfu, ctx_.mem_pool(mem_host), "dm1");
        #pragma omp parallel for
        for (int m = 0; m < nwfu; m++) {
            for (int j = 0; j < kp__.num_occupied_bands(); j++) {
                dm1(j, m) = dm(j, m) * static_cast<T>(kp__.band_occupancy(j, 0));
            }
        }
        if (is_device_memory(mem)) {
            dm1.allocate(ctx_.mem_pool(mem)).copy_to(mem);
        }

        /* now compute O_{ij}^{sigma,sigma'} = \sum_{nk} <psi_nk|phi_{i,sigma}><phi_{j,sigma^'}|psi_nk> f_{nk} */
        auto alpha = std::complex<T>(kp__.weight(), 0.0);
        linalg(la).gemm('C', 'N', nwfu, nwfu, kp__.num_occupied_bands(), &alpha, dm.at(mem), dm.ld(), dm1.at(mem),
                        dm1.ld(), &linalg_const<std::complex<T>>::zero(), occ_mtrx.at(mem), occ_mtrx.ld());
        if (is_device_memory(mem)) {
            occ_mtrx.copy_to(memory_t::host);
        }

        #pragma omp parallel for schedule(static)
        for (int at_lvl = 0; at_lvl < static_cast<int>(local_.size()); at_lvl++) {
            const int ia     = atomic_orbitals_[at_lvl].first;
            auto const& atom = ctx_.unit_cell().atom(ia);
            // we can skip this atomic level if it does not contribute to the Hubbard correction (or U = 0)
            if (atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).use_for_calculation()) {

                /* loop over the different channels */
                /* note that for atom with SO interactions, we need to jump
                   by 2 instead of 1. This is due to the fact that the
                   relativistic wave functions have different total angular
                   momentum for the same n
                */
                const int lmmax_at = 2 * atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).l + 1;
                for (int s1 = 0; s1 < ctx_.num_spins(); s1++) {
                    for (int s2 = 0; s2 < ctx_.num_spins(); s2++) {
                        int s = (s1 == s2) * s1 + (s1 != s2) * (1 + 2 * s2 + s1);
                        for (int mp = 0; mp < lmmax_at; mp++) {
                            for (int m = 0; m < lmmax_at; m++) {
                                local_[at_lvl](m, mp, s) +=
                                    occ_mtrx(r.first * s1 + offset_[at_lvl] + m, r.first * s2 + offset_[at_lvl] + mp);
                            }
                        }
                    }
                }
            }
        }
    } else {
        /* SLDA + U, we need to do the explicit calculation. The hubbard
           orbitals only have one component while the bloch wave functions
           have two. The inner product takes care of this case internally. */

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            PROFILE_START("sirius::Occupation_matrix::add_k_point_contribution|1");
            dmatrix<std::complex<T>> dm(kp__.num_occupied_bands(ispn), nwfu, ctx_.mem_pool(mem_host), "dm");
            if (is_device_memory(mem)) {
                dm.allocate(ctx_.mem_pool(mem));
            }
            /* compute <psi | phi> where |phi> are the Hubbard WFs */
            sddk::inner(ctx_.spla_context(), spin_range(ispn), kp__.spinor_wave_functions(), 0,
                        kp__.num_occupied_bands(ispn), kp__.hubbard_wave_functions_S(), 0, nwfu, dm, 0, 0);
            PROFILE_STOP("sirius::Occupation_matrix::add_k_point_contribution|1");

            PROFILE_START("sirius::Occupation_matrix::add_k_point_contribution|2");
            dmatrix<std::complex<T>> dm1(kp__.num_occupied_bands(ispn), nwfu, ctx_.mem_pool(mem_host), "dm1");
            #pragma omp parallel for
            for (int m = 0; m < nwfu; m++) {
                for (int j = 0; j < kp__.num_occupied_bands(ispn); j++) {
                    dm1(j, m) = dm(j, m) * static_cast<T>(kp__.band_occupancy(j, ispn));
                }
            }
            if (is_device_memory(mem)) {
                dm1.allocate(ctx_.mem_pool(mem)).copy_to(mem);
            }
            PROFILE_STOP("sirius::Occupation_matrix::add_k_point_contribution|2");
            /* now compute O_{ij}^{sigma,sigma'} = \sum_{nk} <psi_nk|phi_{i,sigma}><phi_{j,sigma^'}|psi_nk> f_{nk} */
            /* We need to apply a factor 1/2 when we compute the occupancies for the LDA+U. It is because the
             * calculations of E and U consider occupancies <= 1.  Sirius for the LDA+U has a factor 2 in the
             * band occupancies. We need to compensate for it because it is taken into account in the
             * calculation of the hubbard potential */
            PROFILE_START("sirius::Occupation_matrix::add_k_point_contribution|3");
            auto alpha = std::complex<T>(kp__.weight() / ctx_.max_occupancy(), 0.0);
            linalg(la).gemm('C', 'N', nwfu, nwfu, kp__.num_occupied_bands(ispn), &alpha, dm.at(mem), dm.ld(),
                            dm1.at(mem), dm1.ld(), &linalg_const<std::complex<T>>::zero(), occ_mtrx.at(mem),
                            occ_mtrx.ld());
            if (is_device_memory(mem)) {
                occ_mtrx.copy_to(memory_t::host);
            }
            PROFILE_STOP("sirius::Occupation_matrix::add_k_point_contribution|3");

            PROFILE_START("sirius::Occupation_matrix::add_k_point_contribution|4");
            #pragma omp parallel for schedule(static)
            for (int at_lvl = 0; at_lvl < static_cast<int>(local_.size()); at_lvl++) {
                const int ia     = atomic_orbitals_[at_lvl].first;
                auto const& atom = ctx_.unit_cell().atom(ia);
                // we can skip the symmetrization for this atomic level since it does not contribute to the Hubbard
                // correction (or U = 0)
                if (atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).use_for_calculation()) {

                    const int lmmax_at = 2 * atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).l + 1;
                    for (int mp = 0; mp < lmmax_at; mp++) {
                        const int mmp = offset_[at_lvl] + mp;
                        for (int m = 0; m < lmmax_at; m++) {
                            const int mm = offset_[at_lvl] + m;
                            local_[at_lvl](m, mp, ispn) += occ_mtrx(mm, mmp);
                        }
                    }
                }
            }
            PROFILE_STOP("sirius::Occupation_matrix::add_k_point_contribution|4");

            PROFILE_START("sirius::Occupation_matrix::add_k_point_contribution|nonloc");
            for (auto& e : this->occ_mtrx_T_) {
                /* e^{-i k T} */
                auto z1 = std::exp(double_complex(0, -twopi * dot(e.first, kp__.vk())));
                for (int i = 0; i < nwfu; i++) {
                    for (int j = 0; j < nwfu; j++) {
                        e.second(i, j, ispn) += static_cast<std::complex<T>>(occ_mtrx(i, j)) * z1;
                    }
                }
            }
            PROFILE_STOP("sirius::Occupation_matrix::add_k_point_contribution|nonloc");
        } // ispn
    }
}

template void Occupation_matrix::add_k_point_contribution<double>(K_point<double>& kp__);
#ifdef USE_FP32
template void Occupation_matrix::add_k_point_contribution<float>(K_point<float>& kp__);
#endif

void
Occupation_matrix::symmetrize()
{
    if (!ctx_.hubbard_correction()) {
        return;
    }

    auto& sym      = ctx_.unit_cell().symmetry();
    const double f = 1.0 / sym.size();
    std::vector<sddk::mdarray<double_complex, 3>> local_tmp;

    local_tmp.resize(local_.size());

    for (int at_lvl = 0; at_lvl < static_cast<int>(local_.size()); at_lvl++) {
        const int ia     = atomic_orbitals_[at_lvl].first;
        auto const& atom = ctx_.unit_cell().atom(ia);
        // We can skip the symmetrization for this atomic level since it does not contribute to the Hubbard correction
        // (or U = 0)
        if (atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).use_for_calculation()) {
            local_tmp[at_lvl] = sddk::mdarray<double_complex, 3>(local_[at_lvl].size(0), local_[at_lvl].size(1), 4);
            memcpy(local_tmp[at_lvl].at(memory_t::host), local_[at_lvl].at(memory_t::host),
                   sizeof(double_complex) * local_[at_lvl].size());
        }
    }

    for (int at_lvl = 0; at_lvl < static_cast<int>(local_.size()); at_lvl++) {
        const int ia     = atomic_orbitals_[at_lvl].first;
        const auto& atom = ctx_.unit_cell().atom(ia);

        // we can skip the symmetrization for this atomic level since it does not contribute to the Hubbard correction
        // (or U = 0)
        if (atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).use_for_calculation()) {

            int il             = atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).l;
            const int lmmax_at = 2 * il + 1;
            local_[at_lvl].zero();
            sddk::mdarray<double_complex, 3> dm_ia(lmmax_at, lmmax_at, 4);
            for (int isym = 0; isym < sym.size(); isym++) {
                int pr            = sym[isym].spg_op.proper;
                auto eang         = sym[isym].spg_op.euler_angles;
                auto rotm         = sht::rotation_matrix<double>(4, eang, pr);
                auto spin_rot_su2 = rotation_matrix_su2(sym[isym].spin_rotation);

                int iap = sym[isym].spg_op.inv_sym_atom[ia];
                dm_ia.zero();

                int at_lvl1 =
                    find_orbital_index(iap, atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).n(),
                                       atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).l);

                for (int ispn = 0; ispn < ((ctx_.num_mag_dims() == 3) ? (4) : ctx_.num_spins()); ispn++) {
                    for (int m1 = 0; m1 < lmmax_at; m1++) {
                        for (int m2 = 0; m2 < lmmax_at; m2++) {
                            for (int m1p = 0; m1p < lmmax_at; m1p++) {
                                for (int m2p = 0; m2p < lmmax_at; m2p++) {
                                    dm_ia(m1, m2, ispn) +=
                                        rotm[il](m1, m1p) * rotm[il](m2, m2p) * local_tmp[at_lvl1](m1p, m2p, ispn) * f;
                                }
                            }
                        }
                    }
                }

                if (ctx_.num_mag_dims() == 0) {
                    for (int m1 = 0; m1 < lmmax_at; m1++) {
                        for (int m2 = 0; m2 < lmmax_at; m2++) {
                            local_[at_lvl](m1, m2, 0) += dm_ia(m1, m2, 0);
                        }
                    }
                }

                if (ctx_.num_mag_dims() == 1) {
                    int const map_s[3][2] = {{0, 0}, {1, 1}, {0, 1}};
                    for (int j = 0; j < 2; j++) {
                        int s1 = map_s[j][0];
                        int s2 = map_s[j][1];

                        for (int m1 = 0; m1 < lmmax_at; m1++) {
                            for (int m2 = 0; m2 < lmmax_at; m2++) {
                                double_complex dm[2][2] = {{dm_ia(m1, m2, 0), 0}, {0, dm_ia(m1, m2, 1)}};

                                for (int s1p = 0; s1p < 2; s1p++) {
                                    for (int s2p = 0; s2p < 2; s2p++) {
                                        local_[at_lvl](m1, m2, j) +=
                                            dm[s1p][s2p] * spin_rot_su2(s1, s1p) * std::conj(spin_rot_su2(s2, s2p));
                                    }
                                }
                            }
                        }
                    }
                }

                if (ctx_.num_mag_dims() == 3) {
                    for (int m1 = 0; m1 < lmmax_at; m1++) {
                        for (int m2 = 0; m2 < lmmax_at; m2++) {

                            double_complex dm[2][2];
                            double_complex dm1[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
                            for (int s1 = 0; s1 < ctx_.num_spins(); s1++) {
                                for (int s2 = 0; s2 < ctx_.num_spins(); s2++) {
                                    int s      = (s1 == s2) * s1 + (s1 != s2) * (1 + 2 * s2 + s1);
                                    dm[s1][s2] = dm_ia(m1, m2, s);
                                }
                            }

                            for (int i = 0; i < 2; i++) {
                                for (int j = 0; j < 2; j++) {
                                    for (int s1p = 0; s1p < 2; s1p++) {
                                        for (int s2p = 0; s2p < 2; s2p++) {
                                            dm1[i][j] +=
                                                dm[s1p][s2p] * spin_rot_su2(i, s1p) * std::conj(spin_rot_su2(j, s2p));
                                        }
                                    }
                                }
                            }

                            for (int s1 = 0; s1 < ctx_.num_spins(); s1++) {
                                for (int s2 = 0; s2 < ctx_.num_spins(); s2++) {
                                    int s = (s1 == s2) * s1 + (s1 != s2) * (1 + 2 * s2 + s1);
                                    local_[at_lvl](m1, m2, s) += dm1[s1][s2];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (ctx_.cfg().hubbard().nonlocal().size() && ctx_.num_mag_dims() == 3) {
        RTE_THROW("non-collinear nonlocal occupancy symmetrization is not implemented");
    }

    /* a pair of "total number, offests" for the Hubbard orbitals idexing */
    auto r = ctx_.unit_cell().num_hubbard_wf();

    for (int i = 0; i < static_cast<int>(ctx_.cfg().hubbard().nonlocal().size()); i++) {
        auto nl = ctx_.cfg().hubbard().nonlocal(i);
        int ia  = nl.atom_pair()[0];
        int ja  = nl.atom_pair()[1];
        int il  = nl.l()[0];
        int jl  = nl.l()[1];
        int n1  = nl.n()[0];
        int n2  = nl.n()[1];
        int ib  = 2 * il + 1;
        int jb  = 2 * jl + 1;
        auto T  = nl.T();

        for (int isym = 0; isym < sym.size(); isym++) {
            int pr            = sym[isym].spg_op.proper;
            auto eang         = sym[isym].spg_op.euler_angles;
            auto rotm         = sht::rotation_matrix<double>(4, eang, pr);
            auto spin_rot_su2 = rotation_matrix_su2(sym[isym].spin_rotation);

            int iap = sym[isym].spg_op.inv_sym_atom[ia];
            int jap = sym[isym].spg_op.inv_sym_atom[ja];

            auto Ttot = sym[isym].spg_op.inv_sym_atom_T[ja] - sym[isym].spg_op.inv_sym_atom_T[ia] +
                        dot(sym[isym].spg_op.invR, vector3d<int>(T));

            /* we must search for the right hubbard subspace since we may have
             * multiple orbitals involved in the hubbard correction */
            int at1_lvl    = find_orbital_index(iap, n1, il);
            int at2_lvl    = find_orbital_index(jap, n2, jl);
            auto& occ_mtrx = occ_mtrx_T_[Ttot];

            sddk::mdarray<double_complex, 3> dm_ia_ja(2 * il + 1, 2 * jl + 1, ctx_.num_spins());
            dm_ia_ja.zero();
            /* apply spatial rotation */
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                for (int m1 = 0; m1 < ib; m1++) {
                    for (int m2 = 0; m2 < jb; m2++) {
                        for (int m1p = 0; m1p < ib; m1p++) {
                            for (int m2p = 0; m2p < jb; m2p++) {
                                dm_ia_ja(m1, m2, ispn) +=
                                    rotm[il](m1, m1p) * rotm[jl](m2, m2p) *
                                    occ_mtrx(offset_[at1_lvl] + m1p, offset_[at2_lvl] + m2p, ispn) * f;
                            }
                        }
                    }
                }
            }

            if (ctx_.num_mag_dims() == 0) {
                for (int m1 = 0; m1 < ib; m1++) {
                    for (int m2 = 0; m2 < jb; m2++) {
                        nonlocal_[i](m1, m2, 0) += dm_ia_ja(m1, m2, 0);
                    }
                }
            }
            if (ctx_.num_mag_dims() == 1) {
                int const map_s[3][2] = {{0, 0}, {1, 1}, {0, 1}};
                for (int j = 0; j < 2; j++) {
                    int s1 = map_s[j][0];
                    int s2 = map_s[j][1];

                    for (int m1 = 0; m1 < ib; m1++) {
                        for (int m2 = 0; m2 < jb; m2++) {
                            double_complex dm[2][2] = {{dm_ia_ja(m1, m2, 0), 0}, {0, dm_ia_ja(m1, m2, 1)}};

                            for (int s1p = 0; s1p < 2; s1p++) {
                                for (int s2p = 0; s2p < 2; s2p++) {
                                    nonlocal_[i](m1, m2, j) +=
                                        dm[s1p][s2p] * spin_rot_su2(s1, s1p) * std::conj(spin_rot_su2(s2, s2p));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
void
Occupation_matrix::init()
{
    if (!ctx_.hubbard_correction()) {
        return;
    }

    this->zero();
    #pragma omp parallel for schedule(static)
    for (int at_lvl = 0; at_lvl < static_cast<int>(local_.size()); at_lvl++) {
        const int ia     = atomic_orbitals_[at_lvl].first;
        auto const& atom = ctx_.unit_cell().atom(ia);

        if (atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).use_for_calculation()) {

            int il            = atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).l;
            const int lmax_at = 2 * il + 1;
            if (atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).initial_occupancy.size()) {
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    for (int m = 0; m < lmax_at; m++) {
                        this->local_[at_lvl](m, m, ispn) = atom.type()
                                                               .lo_descriptor_hub(atomic_orbitals_[at_lvl].second)
                                                               .initial_occupancy[m + ispn * lmax_at];
                    }
                }
            } else {
                // compute the total charge for the hubbard orbitals
                double charge = atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).occupancy();
                bool nm       = true; // true if the atom is non magnetic
                int majs, mins;
                if (ctx_.num_spins() != 1) {
                    if (atom.vector_field()[2] > 0.0) {
                        nm   = false;
                        majs = 0;
                        mins = 1;
                    } else if (atom.vector_field()[2] < 0.0) {
                        nm   = false;
                        majs = 1;
                        mins = 0;
                    }
                }

                if (!nm) {
                    if (ctx_.num_mag_dims() != 3) {
                        // collinear case
                        if (charge > (lmax_at)) {
                            for (int m = 0; m < lmax_at; m++) {
                                this->local_[at_lvl](m, m, majs) = 1.0;
                                this->local_[at_lvl](m, m, mins) =
                                    (charge - static_cast<double>(lmax_at)) / static_cast<double>(lmax_at);
                            }
                        } else {
                            for (int m = 0; m < lmax_at; m++) {
                                this->local_[at_lvl](m, m, majs) = charge / static_cast<double>(lmax_at);
                            }
                        }
                    } else {
                        // double c1, s1;
                        // sincos(atom.type().starting_magnetization_theta(), &s1, &c1);
                        double c1 = atom.vector_field()[2];
                        double_complex cs =
                            double_complex(atom.vector_field()[0], atom.vector_field()[1]) / sqrt(1.0 - c1 * c1);
                        double_complex ns[4];

                        if (charge > (lmax_at)) {
                            ns[majs] = 1.0;
                            ns[mins] = (charge - static_cast<double>(lmax_at)) / static_cast<double>(lmax_at);
                        } else {
                            ns[majs] = charge / static_cast<double>(lmax_at);
                            ns[mins] = 0.0;
                        }

                        // charge and moment
                        double nc  = ns[majs].real() + ns[mins].real();
                        double mag = ns[majs].real() - ns[mins].real();

                        // rotate the occ matrix
                        ns[0] = (nc + mag * c1) * 0.5;
                        ns[1] = (nc - mag * c1) * 0.5;
                        ns[2] = mag * std::conj(cs) * 0.5;
                        ns[3] = mag * cs * 0.5;

                        for (int m = 0; m < lmax_at; m++) {
                            this->local_[at_lvl](m, m, 0) = ns[0];
                            this->local_[at_lvl](m, m, 1) = ns[1];
                            this->local_[at_lvl](m, m, 2) = ns[2];
                            this->local_[at_lvl](m, m, 3) = ns[3];
                        }
                    }
                } else {
                    for (int s = 0; s < ctx_.num_spins(); s++) {
                        for (int m = 0; m < lmax_at; m++) {
                            this->local_[at_lvl](m, m, s) = charge * 0.5 / static_cast<double>(lmax_at);
                        }
                    }
                }
            }
        }
    }

    print_occupancies(2);
}

void
Occupation_matrix::print_occupancies(int verbosity__) const
{
    if (!ctx_.hubbard_correction()) {
        return;
    }

    if ((ctx_.verbosity() >= verbosity__) && (ctx_.comm().rank() == 0)) {
        std::stringstream s;
        for (int at_lvl = 0; at_lvl < static_cast<int>(local_.size()); at_lvl++) {
            auto const& atom = ctx_.unit_cell().atom(atomic_orbitals_[at_lvl].first);
            int il           = atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).l;
            if (atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).use_for_calculation()) {
                Hubbard_matrix::print_local(at_lvl, s);
                double occ[2] = {0, 0};
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    for (int m = 0; m < 2 * il + 1; m++) {
                        occ[ispn] += this->local_[at_lvl](m, m, ispn).real();
                    }
                }
                if (ctx_.num_spins() == 2) {
                    s << "Atom charge (total) " << occ[0] + occ[1] << " (n_up) " << occ[0] << " (n_down) " << occ[1]
                      << " (mz) " << occ[0] - occ[1] << std::endl;
                } else {
                    s << "Atom charge (total) " << 2 * occ[0] << std::endl;
                }
            }
        }
        if (ctx_.cfg().hubbard().nonlocal().size()) {
            s << std::endl;
        }
        for (int i = 0; i < static_cast<int>(ctx_.cfg().hubbard().nonlocal().size()); i++) {
            Hubbard_matrix::print_nonlocal(i, s);
        }
        ctx_.message(1, "occ.mtrx", s);
    }
}

} // namespace sirius
