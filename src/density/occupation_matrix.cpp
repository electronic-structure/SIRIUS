/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file occupation_matrix.cpp
 *
 *  \brief Occupation matrix of the LDA+U method.
 */

#include "occupation_matrix.hpp"
#include "symmetry/crystal_symmetry.hpp"
#include "symmetry/symmetrize_occupation_matrix.hpp"

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
                        dot(sym[isym].spg_op.invR, r3::vector<int>(T));
            if (!occ_mtrx_T_.count(Ttot)) {
                occ_mtrx_T_[Ttot] = mdarray<std::complex<double>, 3>({nhwf, nhwf, ctx_.num_mag_comp()});
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

    PROFILE("sirius::Occupation_matrix::add_k_point_contribution");

    memory_t mem_host{memory_t::host};
    memory_t mem{memory_t::host};
    la::lib_t la{la::lib_t::blas};
    if (ctx_.processing_unit() == device_t::GPU) {
        mem      = memory_t::device;
        mem_host = memory_t::host_pinned;
        la       = la::lib_t::gpublas;
    }

    /* a pair of "total number, offests" for the Hubbard orbitals idexing */
    auto r = ctx_.unit_cell().num_hubbard_wf();

    int nwfu = r.first;

    matrix<std::complex<T>> occ_mtrx({nwfu, nwfu}, get_memory_pool(memory_t::host), mdarray_label("occ_mtrx"));
    if (is_device_memory(mem)) {
        occ_mtrx.allocate(get_memory_pool(mem));
    }

    // TODO colinear and non-collinear cases have a lot of similar code; there should be a way to combine it

    /* full non collinear magnetism */
    if (ctx_.num_mag_dims() == 3) {
        la::dmatrix<std::complex<T>> dm(kp__.num_occupied_bands(), nwfu, get_memory_pool(mem_host), "dm");
        if (is_device_memory(mem)) {
            dm.allocate(get_memory_pool(mem));
        }
        wf::inner(ctx_.spla_context(), mem, wf::spin_range(0, 2), kp__.spinor_wave_functions(),
                  wf::band_range(0, kp__.num_occupied_bands()), kp__.hubbard_wave_functions_S(),
                  wf::band_range(0, nwfu), dm, 0, 0);

        la::dmatrix<std::complex<T>> dm1(kp__.num_occupied_bands(), nwfu, get_memory_pool(mem_host), "dm1");
        #pragma omp parallel for
        for (int m = 0; m < nwfu; m++) {
            for (int j = 0; j < kp__.num_occupied_bands(); j++) {
                dm1(j, m) = dm(j, m) * static_cast<T>(kp__.band_occupancy(j, 0));
            }
        }
        if (is_device_memory(mem)) {
            dm1.allocate(get_memory_pool(mem)).copy_to(mem);
        }

        /* now compute O_{ij}^{sigma,sigma'} = \sum_{nk} <psi_nk|phi_{i,sigma}><phi_{j,sigma^'}|psi_nk> f_{nk} */
        auto alpha = std::complex<T>(kp__.weight(), 0.0);
        la::wrap(la).gemm('C', 'N', nwfu, nwfu, kp__.num_occupied_bands(), &alpha, dm.at(mem), dm.ld(), dm1.at(mem),
                          dm1.ld(), &la::constant<std::complex<T>>::zero(), occ_mtrx.at(mem), occ_mtrx.ld());
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
                int s_idx[2][2]    = {{0, 3}, {2, 1}};
                const int lmmax_at = 2 * atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).l() + 1;
                for (int s1 = 0; s1 < ctx_.num_spins(); s1++) {
                    for (int s2 = 0; s2 < ctx_.num_spins(); s2++) {
                        for (int mp = 0; mp < lmmax_at; mp++) {
                            for (int m = 0; m < lmmax_at; m++) {
                                local_[at_lvl](m, mp, s_idx[s1][s2]) += occ_mtrx(r.first * s1 + offset_[at_lvl] + m,
                                                                                 r.first * s2 + offset_[at_lvl] + mp);
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
            if (kp__.num_occupied_bands(ispn) == 0) {
                continue;
            }
            la::dmatrix<std::complex<T>> dm(kp__.num_occupied_bands(ispn), nwfu, get_memory_pool(mem_host), "dm");
            if (is_device_memory(mem)) {
                dm.allocate(get_memory_pool(mem));
            }
            /* compute <psi | phi> where |phi> are the Hubbard WFs */
            wf::inner(ctx_.spla_context(), mem, wf::spin_range(ispn), kp__.spinor_wave_functions(),
                      wf::band_range(0, kp__.num_occupied_bands(ispn)), kp__.hubbard_wave_functions_S(),
                      wf::band_range(0, nwfu), dm, 0, 0);

            la::dmatrix<std::complex<T>> dm1(kp__.num_occupied_bands(ispn), nwfu, get_memory_pool(mem_host), "dm1");
            #pragma omp parallel for
            for (int m = 0; m < nwfu; m++) {
                for (int j = 0; j < kp__.num_occupied_bands(ispn); j++) {
                    dm1(j, m) = dm(j, m) * static_cast<T>(kp__.band_occupancy(j, ispn));
                }
            }
            if (is_device_memory(mem)) {
                dm1.allocate(get_memory_pool(mem)).copy_to(mem);
            }
            /* now compute O_{ij}^{sigma,sigma'} = \sum_{nk} <psi_nk|phi_{i,sigma}><phi_{j,sigma^'}|psi_nk> f_{nk} */
            /* We need to apply a factor 1/2 when we compute the occupancies for the LDA+U. It is because the
             * calculations of E and U consider occupancies <= 1.  Sirius for the LDA+U has a factor 2 in the
             * band occupancies. We need to compensate for it because it is taken into account in the
             * calculation of the hubbard potential */
            auto alpha = std::complex<T>(kp__.weight() / ctx_.max_occupancy(), 0.0);
            la::wrap(la).gemm('C', 'N', nwfu, nwfu, kp__.num_occupied_bands(ispn), &alpha, dm.at(mem), dm.ld(),
                              dm1.at(mem), dm1.ld(), &la::constant<std::complex<T>>::zero(), occ_mtrx.at(mem),
                              occ_mtrx.ld());
            if (is_device_memory(mem)) {
                occ_mtrx.copy_to(memory_t::host);
            }

            #pragma omp parallel for
            for (int at_lvl = 0; at_lvl < static_cast<int>(local_.size()); at_lvl++) {
                const int ia     = atomic_orbitals_[at_lvl].first;
                auto const& atom = ctx_.unit_cell().atom(ia);
                // we can skip the symmetrization for this atomic level if it does not contribute to the Hubbard
                // correction (or U = 0)
                if (atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).use_for_calculation()) {

                    const int lmmax_at = 2 * atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).l() + 1;
                    for (int mp = 0; mp < lmmax_at; mp++) {
                        const int mmp = offset_[at_lvl] + mp;
                        for (int m = 0; m < lmmax_at; m++) {
                            const int mm = offset_[at_lvl] + m;
                            local_[at_lvl](m, mp, ispn) += occ_mtrx(mm, mmp);
                        }
                    }
                }
            }

            for (auto& e : this->occ_mtrx_T_) {
                /* e^{-i k T} */
                auto z1 = std::exp(std::complex<double>(0, -twopi * dot(e.first, kp__.vk())));
                for (int i = 0; i < nwfu; i++) {
                    for (int j = 0; j < nwfu; j++) {
                        e.second(i, j, ispn) +=
                                static_cast<std::complex<T>>(occ_mtrx(i, j)) * static_cast<std::complex<T>>(z1);
                    }
                }
            }
        } // ispn
    }
}

template void
Occupation_matrix::add_k_point_contribution<double>(K_point<double>& kp__);
#ifdef SIRIUS_USE_FP32
template void
Occupation_matrix::add_k_point_contribution<float>(K_point<float>& kp__);
#endif

void
Occupation_matrix::init()
{
    if (!ctx_.hubbard_correction()) {
        return;
    }

    this->zero();
    #pragma omp parallel for schedule(static)
    for (int at_lvl = 0; at_lvl < static_cast<int>(local_.size()); at_lvl++) {
        const int ia      = atomic_orbitals_[at_lvl].first;
        auto const& atom  = ctx_.unit_cell().atom(ia);
        const int il      = atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).l();
        const int lmax_at = 2 * il + 1;

        if (atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).use_for_calculation()) {
            if (atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).initial_occupancy().size()) {
                /* if we specify the occcupancy in the input file */
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    for (int m = 0; m < lmax_at; m++) {
                        this->local_[at_lvl](m, m, ispn) = atom.type()
                                                                   .lo_descriptor_hub(atomic_orbitals_[at_lvl].second)
                                                                   .initial_occupancy()[m + ispn * lmax_at];
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
                        double c1               = atom.vector_field()[2];
                        std::complex<double> cs = std::complex<double>(atom.vector_field()[0], atom.vector_field()[1]) /
                                                  sqrt(1.0 - c1 * c1);
                        std::complex<double> ns[4];

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
            // initialize the occupancy matrices to their user provided values.
            if (ctx_.cfg().hubbard().constrained_calculation() && apply_constraints_.size()) {
                if (apply_constraints_[at_lvl]) {
                    copy(local_constraints_[at_lvl], local_[at_lvl]);
                }
            }
        }
    }
    print_occupancies(2);
}

void
Occupation_matrix::calculate_constraints_and_error()
{
    if (apply_constraint()) {
        double error_ = 0.0;
        for (int at_lvl = 0; at_lvl < static_cast<int>(local_.size()); at_lvl++) {
            if (apply_constraints_[at_lvl]) {
                const int ia      = atomic_orbitals_[at_lvl].first;
                auto const& atom  = ctx_.unit_cell().atom(ia);
                int il            = atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).l();
                const int lmax_at = 2 * il + 1;
                for (int is = 0; is < ctx_.num_spins(); is++) {
                    for (int m1 = 0; m1 < lmax_at; m1++) {
                        for (int m2 = 0; m2 < lmax_at; m2++) {
                            std::complex<double> tmp =
                                    this->local_[at_lvl](m2, m1, is) - this->local_constraints_[at_lvl](m2, m1, is);
                            multipliers_constraints_[at_lvl](m2, m1, is) +=
                                    tmp * ctx_.cfg().hubbard().constraint_beta_mixing();
                            error_ = std::max(error_, std::abs(tmp));
                        }
                    }
                }
            }
        }
        this->constraint_error_ = error_;
        this->num_steps_++;
    }
}

void
Occupation_matrix::print_occupancies(int verbosity__) const
{
    if (!ctx_.hubbard_correction()) {
        return;
    }

    if (ctx_.comm().rank() == 0) {
        std::stringstream s;
        /* print local part */
        if (ctx_.verbosity() >= verbosity__) {
            for (int at_lvl = 0; at_lvl < static_cast<int>(local_.size()); at_lvl++) {
                auto const& atom = ctx_.unit_cell().atom(atomic_orbitals_[at_lvl].first);
                int il           = atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).l();
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
        }
        /* print non-local part */
        if (ctx_.cfg().hubbard().nonlocal().size() && (ctx_.verbosity() >= verbosity__ + 1)) {
            s << std::endl;
            for (int i = 0; i < static_cast<int>(ctx_.cfg().hubbard().nonlocal().size()); i++) {
                Hubbard_matrix::print_nonlocal(i, s);
            }
        }
        if (ctx_.verbosity() >= verbosity__) {
            ctx_.message(1, "occ.mtrx", s);
        }
    }
}

} // namespace sirius
