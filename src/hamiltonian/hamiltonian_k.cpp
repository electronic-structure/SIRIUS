/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file hamiltonian_k.cpp
 *
 *  \brief Contains definition of sirius::Hamiltonian_k class.
 */

#include "context/simulation_context.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "hamiltonian/local_operator.hpp"
#include "hamiltonian/non_local_operator.hpp"
#include "potential/potential.hpp"
#include "core/wf/wave_functions.hpp"
#include "core/omp.hpp"
#include "core/profiler.hpp"
#include "k_point/k_point.hpp"
#include "lapw/generate_alm_block.hpp"
#include <chrono>

namespace sirius {

template <typename T>
Hamiltonian_k<T>::Hamiltonian_k(Hamiltonian0<T> const& H0__, K_point<T>& kp__)
    : H0_(H0__)
    , kp_(kp__)
{
    PROFILE("sirius::Hamiltonian_k");
    H0_.local_op().prepare_k(kp_.gkvec_fft());
    if (!H0_.ctx().full_potential()) {
        if (H0_.ctx().hubbard_correction()) {
            u_op_ = std::make_shared<U_operator<T>>(H0__.ctx(), H0__.potential().hubbard_potential(), kp__.vk());
            if (H0_.ctx().processing_unit() == device_t::GPU) {
                const_cast<wf::Wave_functions<T>&>(kp_.hubbard_wave_functions_S())
                        .allocate(H0_.ctx().processing_unit_memory_t());
                const_cast<wf::Wave_functions<T>&>(kp_.hubbard_wave_functions_S())
                        .copy_to(H0_.ctx().processing_unit_memory_t());
            }
        }
    }
}

template <typename T>
Hamiltonian_k<T>::~Hamiltonian_k()
{
    if (!H0_.ctx().full_potential()) {
        if (H0_.ctx().hubbard_correction()) {
            if (H0_.ctx().processing_unit() == device_t::GPU) {
                const_cast<wf::Wave_functions<T>&>(kp_.hubbard_wave_functions_S())
                        .deallocate(H0_.ctx().processing_unit_memory_t());
            }
        }
    }
}

template <typename T>
Hamiltonian_k<T>::Hamiltonian_k(Hamiltonian_k&& src__) = default;

template <typename T>
template <typename F, int what>
std::pair<mdarray<T, 2>, mdarray<T, 2>>
Hamiltonian_k<T>::get_h_o_diag_pw() const
{
    PROFILE("sirius::Hamiltonian_k::get_h_o_diag");

    auto const& uc = H0_.ctx().unit_cell();

    mdarray<T, 2> h_diag({kp_.num_gkvec_loc(), H0_.ctx().num_spins()});
    mdarray<T, 2> o_diag({kp_.num_gkvec_loc(), H0_.ctx().num_spins()});

    h_diag.zero();
    o_diag.zero();

    std::vector<int> offset_t(uc.num_atom_types());
    std::generate(offset_t.begin(), offset_t.end(), [n = 0, iat = 0, &uc]() mutable {
        int offs = n;
        n += uc.atom_type(iat++).mt_basis_size();
        return offs;
    });

    for (int ispn = 0; ispn < H0_.ctx().num_spins(); ispn++) {

        /* local H contribution */
        #pragma omp parallel for schedule(static)
        for (int ig_loc = 0; ig_loc < kp_.num_gkvec_loc(); ig_loc++) {
            if (what & 1) {
                auto ekin            = 0.5 * kp_.gkvec().gkvec_cart(gvec_index_t::local(ig_loc)).length2();
                h_diag(ig_loc, ispn) = ekin + H0_.local_op().v0(ispn);
            }
            if (what & 2) {
                o_diag(ig_loc, ispn) = 1;
            }
        }
        if (uc.max_mt_basis_size() == 0) {
            continue;
        }

        /* non-local H contribution */
        auto beta_gk_t = kp_.beta_projectors().pw_coeffs_t(0);
        matrix<std::complex<T>> beta_gk_tmp({kp_.num_gkvec_loc(), uc.max_mt_basis_size()});

        for (int iat = 0; iat < uc.num_atom_types(); iat++) {
            auto& atom_type = uc.atom_type(iat);
            int nbf         = atom_type.mt_basis_size();
            if (!nbf) {
                continue;
            }

            matrix<std::complex<T>> d_sum;
            if (what & 1) {
                d_sum = matrix<std::complex<T>>({nbf, nbf});
                d_sum.zero();
            }

            matrix<std::complex<T>> q_sum;
            if (what & 2) {
                q_sum = matrix<std::complex<T>>({nbf, nbf});
                q_sum.zero();
            }

            for (int i = 0; i < atom_type.num_atoms(); i++) {
                int ia = atom_type.atom_id(i);

                for (int xi2 = 0; xi2 < nbf; xi2++) {
                    for (int xi1 = 0; xi1 < nbf; xi1++) {
                        if (what & 1) {
                            d_sum(xi1, xi2) += H0_.D().template value<F>(xi1, xi2, ispn, ia);
                        }
                        if (what & 2) {
                            q_sum(xi1, xi2) += H0_.Q().template value<F>(xi1, xi2, ispn, ia);
                        }
                    }
                }
            }

            int offs = offset_t[iat];

            if (what & 1) {
                la::wrap(la::lib_t::blas)
                        .gemm('N', 'N', kp_.num_gkvec_loc(), nbf, nbf, &la::constant<std::complex<T>>::one(),
                              &beta_gk_t(0, offs), beta_gk_t.ld(), &d_sum(0, 0), d_sum.ld(),
                              &la::constant<std::complex<T>>::zero(), &beta_gk_tmp(0, 0), beta_gk_tmp.ld());
                #pragma omp parallel
                for (int xi = 0; xi < nbf; xi++) {
                    #pragma omp for schedule(static) nowait
                    for (int ig_loc = 0; ig_loc < kp_.num_gkvec_loc(); ig_loc++) {
                        /* compute <G+k|beta_xi1> D_{xi1, xi2} <beta_xi2|G+k> contribution from all atoms */
                        h_diag(ig_loc, ispn) +=
                                std::real(beta_gk_tmp(ig_loc, xi) * std::conj(beta_gk_t(ig_loc, offs + xi)));
                    }
                }
            }

            if (what & 2) {
                la::wrap(la::lib_t::blas)
                        .gemm('N', 'N', kp_.num_gkvec_loc(), nbf, nbf, &la::constant<std::complex<T>>::one(),
                              &beta_gk_t(0, offs), beta_gk_t.ld(), &q_sum(0, 0), q_sum.ld(),
                              &la::constant<std::complex<T>>::zero(), &beta_gk_tmp(0, 0), beta_gk_tmp.ld());
                #pragma omp parallel
                for (int xi = 0; xi < nbf; xi++) {
                    #pragma omp for schedule(static) nowait
                    for (int ig_loc = 0; ig_loc < kp_.num_gkvec_loc(); ig_loc++) {
                        /* compute <G+k|beta_xi1> Q_{xi1, xi2} <beta_xi2|G+k> contribution from all atoms */
                        o_diag(ig_loc, ispn) +=
                                std::real(beta_gk_tmp(ig_loc, xi) * std::conj(beta_gk_t(ig_loc, offs + xi)));
                    }
                }
            }
        }
    }
    if (H0_.ctx().processing_unit() == device_t::GPU) {
        if (what & 1) {
            h_diag.allocate(memory_t::device).copy_to(memory_t::device);
        }
        if (what & 2) {
            o_diag.allocate(memory_t::device).copy_to(memory_t::device);
        }
    }
    return std::make_pair(std::move(h_diag), std::move(o_diag));
}

template <typename T>
template <int what>
std::pair<mdarray<T, 2>, mdarray<T, 2>>
Hamiltonian_k<T>::get_h_o_diag_lapw() const
{
    PROFILE("sirius::Hamiltonian::get_h_o_diag");

    auto const& uc = H0_.ctx().unit_cell();

    splindex_block<atom_index_t> spl_num_atoms(uc.num_atoms(), n_blocks(kp_.comm().size()),
                                               block_id(kp_.comm().rank()));
    int nlo{0};
    for (auto it : spl_num_atoms) {
        nlo += uc.atom(it.i).mt_lo_basis_size();
    }

    auto h_diag = (what & 1) ? mdarray<T, 2>({kp_.num_gkvec_loc() + nlo, 1}) : mdarray<T, 2>();
    auto o_diag = (what & 2) ? mdarray<T, 2>({kp_.num_gkvec_loc() + nlo, 1}) : mdarray<T, 2>();

    #pragma omp parallel for schedule(static)
    for (int igloc = 0; igloc < kp_.num_gkvec_loc(); igloc++) {
        if (what & 1) {
            auto gvc      = kp_.gkvec().gkvec_cart(gvec_index_t::local(igloc));
            T ekin        = 0.5 * dot(gvc, gvc);
            h_diag[igloc] = H0_.local_op().v0(0) + ekin * H0_.ctx().theta_pw(0).real();
        }
        if (what & 2) {
            o_diag[igloc] = H0_.ctx().theta_pw(0).real();
        }
    }

    #pragma omp parallel
    {
        matrix<std::complex<T>> alm({kp_.num_gkvec_loc(), uc.max_mt_aw_basis_size()});

        auto halm = (what & 1) ? matrix<std::complex<T>>({kp_.num_gkvec_loc(), uc.max_mt_aw_basis_size()})
                               : matrix<std::complex<T>>();

        auto h_diag_omp = (what & 1) ? mdarray<T, 1>({kp_.num_gkvec_loc()}) : mdarray<T, 1>();
        if (what & 1) {
            h_diag_omp.zero();
        }

        auto o_diag_omp = (what & 2) ? mdarray<T, 1>({kp_.num_gkvec_loc()}) : mdarray<T, 1>();
        if (what & 2) {
            o_diag_omp.zero();
        }

        #pragma omp for
        for (int ia = 0; ia < uc.num_atoms(); ia++) {
            auto& atom = uc.atom(ia);
            int nmt    = atom.mt_aw_basis_size();

            kp_.alm_coeffs_loc().template generate<false>(atom, alm);
            if (what & 1) {
                H0_.apply_hmt_to_apw(atom, spin_block_t::nm, kp_.num_gkvec_loc(), alm, halm);
            }

            for (int xi = 0; xi < nmt; xi++) {
                for (int igloc = 0; igloc < kp_.num_gkvec_loc(); igloc++) {
                    if (what & 1) {
                        h_diag_omp[igloc] += std::real(std::conj(alm(igloc, xi)) * halm(igloc, xi));
                    }
                    if (what & 2) {
                        o_diag_omp[igloc] += std::real(std::conj(alm(igloc, xi)) * alm(igloc, xi));
                    }
                }
            }
        }

        #pragma omp critical
        for (int igloc = 0; igloc < kp_.num_gkvec_loc(); igloc++) {
            if (what & 1) {
                h_diag[igloc] += h_diag_omp[igloc];
            }
            if (what & 2) {
                o_diag[igloc] += o_diag_omp[igloc];
            }
        }
    }

    nlo = 0;
    for (auto it : spl_num_atoms) {
        auto& atom = uc.atom(it.i);
        auto& type = atom.type();
        auto& hmt  = H0_.hmt(it.i);
        #pragma omp parallel for
        for (int ilo = 0; ilo < type.mt_lo_basis_size(); ilo++) {
            int xi_lo = type.mt_aw_basis_size() + ilo;
            if (what & 1) {
                h_diag[kp_.num_gkvec_loc() + nlo + ilo] = hmt(xi_lo, xi_lo).real();
            }
            if (what & 2) {
                o_diag[kp_.num_gkvec_loc() + nlo + ilo] = 1;
            }
        }
        nlo += atom.mt_lo_basis_size();
    }

    if (H0_.ctx().processing_unit() == device_t::GPU) {
        if (what & 1) {
            h_diag.allocate(memory_t::device).copy_to(memory_t::device);
        }
        if (what & 2) {
            o_diag.allocate(memory_t::device).copy_to(memory_t::device);
        }
    }
    return std::make_pair(std::move(h_diag), std::move(o_diag));
}

template <typename T>
void
Hamiltonian_k<T>::set_fv_h_o(la::dmatrix<std::complex<T>>& h__, la::dmatrix<std::complex<T>>& o__) const
{
    PROFILE("sirius::Hamiltonian_k::set_fv_h_o");

    /* alias to unit cell */
    auto& uc = H0_.ctx().unit_cell();
    /* alias to k-point */
    /* split atoms in blocks */
    int num_atoms_in_block = 2 * omp_get_max_threads();
    int nblk               = uc.num_atoms() / num_atoms_in_block + std::min(1, uc.num_atoms() % num_atoms_in_block);

    // TODO: use new way to split in blocks
    // TODO: use generate_alm_block()

    /* maximum number of apw coefficients in the block of atoms */
    int max_mt_aw = num_atoms_in_block * uc.max_mt_aw_basis_size();
    /* current processing unit */
    auto pu = H0_.ctx().processing_unit();

    auto la  = la::lib_t::none;
    auto mt  = memory_t::none;
    auto mt1 = memory_t::none;
    int nb   = 0;
    switch (pu) {
        case device_t::CPU: {
            la  = la::lib_t::blas;
            mt  = memory_t::host;
            mt1 = memory_t::host;
            nb  = 1;
            break;
        }
        case device_t::GPU: {
            la  = la::lib_t::spla;
            mt  = memory_t::host_pinned;
            mt1 = memory_t::device;
            nb  = 1;
            break;
        }
    }

    mdarray<std::complex<T>, 3> alm_row({kp_.num_gkvec_row(), max_mt_aw, nb}, get_memory_pool(mt));
    mdarray<std::complex<T>, 3> alm_col({kp_.num_gkvec_col(), max_mt_aw, nb}, get_memory_pool(mt));
    mdarray<std::complex<T>, 3> halm_col({kp_.num_gkvec_col(), max_mt_aw, nb}, get_memory_pool(mt));

    print_memory_usage(H0_.ctx().out(), FILE_LINE);

    h__.zero();
    o__.zero();
    switch (pu) {
        case device_t::GPU: {
            alm_row.allocate(get_memory_pool(memory_t::device));
            alm_col.allocate(get_memory_pool(memory_t::device));
            halm_col.allocate(get_memory_pool(memory_t::device));
            break;
        }
        case device_t::CPU: {
            break;
        }
    }

    /* offsets for matching coefficients of individual atoms in the AW block */
    std::vector<int> offsets(uc.num_atoms());

    PROFILE_START("sirius::Hamiltonian_k::set_fv_h_o|zgemm");
    const auto t1 = time_now();
    /* loop over blocks of atoms */
    for (int iblk = 0; iblk < nblk; iblk++) {
        /* number of matching AW coefficients in the block */
        int num_mt_aw{0};
        int ia_begin = iblk * num_atoms_in_block;
        int ia_end   = std::min(uc.num_atoms(), (iblk + 1) * num_atoms_in_block);
        for (int ia = ia_begin; ia < ia_end; ia++) {
            offsets[ia] = num_mt_aw;
            num_mt_aw += uc.atom(ia).type().mt_aw_basis_size();
        }

        int s = (pu == device_t::GPU) ? (iblk % 2) : 0;
        s     = 0;

        if (env::print_checksum()) {
            alm_row.zero();
            alm_col.zero();
            halm_col.zero();
        }

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for
            for (int ia = ia_begin; ia < ia_end; ia++) {
                auto& atom = uc.atom(ia);
                auto& type = atom.type();
                int naw    = type.mt_aw_basis_size();

                mdarray<std::complex<T>, 2> alm_row_atom;
                mdarray<std::complex<T>, 2> alm_col_atom;
                mdarray<std::complex<T>, 2> halm_col_atom;

                switch (pu) {
                    case device_t::CPU: {
                        alm_row_atom = mdarray<std::complex<T>, 2>({kp_.num_gkvec_row(), naw},
                                                                   alm_row.at(memory_t::host, 0, offsets[ia], s));

                        alm_col_atom = mdarray<std::complex<T>, 2>({kp_.num_gkvec_col(), naw},
                                                                   alm_col.at(memory_t::host, 0, offsets[ia], s));

                        halm_col_atom = mdarray<std::complex<T>, 2>({kp_.num_gkvec_col(), naw},
                                                                    halm_col.at(memory_t::host, 0, offsets[ia], s));
                        break;
                    }
                    case device_t::GPU: {
                        alm_row_atom = mdarray<std::complex<T>, 2>({kp_.num_gkvec_row(), naw},
                                                                   alm_row.at(memory_t::host, 0, offsets[ia], s),
                                                                   alm_row.at(memory_t::device, 0, offsets[ia], s));

                        alm_col_atom = mdarray<std::complex<T>, 2>({kp_.num_gkvec_col(), naw},
                                                                   alm_col.at(memory_t::host, 0, offsets[ia], s),
                                                                   alm_col.at(memory_t::device, 0, offsets[ia], s));

                        halm_col_atom = mdarray<std::complex<T>, 2>({kp_.num_gkvec_col(), naw},
                                                                    halm_col.at(memory_t::host, 0, offsets[ia], s),
                                                                    halm_col.at(memory_t::device, 0, offsets[ia], s));
                        break;
                    }
                }

                kp_.alm_coeffs_col().template generate<false>(atom, alm_col_atom);
                /* can't copy alm to device how as it might be modified by the iora */

                H0_.apply_hmt_to_apw(atom, spin_block_t::nm, kp_.num_gkvec_col(), alm_col_atom, halm_col_atom);
                if (pu == device_t::GPU) {
                    halm_col_atom.copy_to(memory_t::device, acc::stream_id(tid));
                }

                /* generate conjugated matching coefficients */
                kp_.alm_coeffs_row().template generate<true>(atom, alm_row_atom);
                if (pu == device_t::GPU) {
                    alm_row_atom.copy_to(memory_t::device, acc::stream_id(tid));
                }

                /* setup apw-lo and lo-apw blocks */
                set_fv_h_o_apw_lo(atom, ia, alm_row_atom, alm_col_atom, h__, o__);

                /* finally, modify alm coefficients for iora */
                if (H0_.ctx().valence_relativity() == relativity_t::iora) {
                    // TODO: check if we can modify alm_col with IORA eralier and then not apply it in
                    // set_fv_h_o_apw_lo()
                    H0_.add_o1mt_to_apw(atom, kp_.num_gkvec_col(), alm_col_atom);
                }

                if (pu == device_t::GPU) {
                    alm_col_atom.copy_to(memory_t::device, acc::stream_id(tid));
                }
            }
            acc::sync_stream(acc::stream_id(tid));
        }
        // acc::sync_stream(stream_id(omp_get_max_threads()));

        if (env::print_checksum()) {
            auto z1 = alm_row.checksum();
            auto z2 = alm_col.checksum();
            auto z3 = halm_col.checksum();
            print_checksum("alm_row", z1, H0_.ctx().out());
            print_checksum("alm_col", z2, H0_.ctx().out());
            print_checksum("halm_col", z3, H0_.ctx().out());
        }

        la::wrap(la).gemm('N', 'T', kp_.num_gkvec_row(), kp_.num_gkvec_col(), num_mt_aw,
                          &la::constant<std::complex<T>>::one(), alm_row.at(mt1, 0, 0, s), alm_row.ld(),
                          alm_col.at(mt1, 0, 0, s), alm_col.ld(), &la::constant<std::complex<T>>::one(), o__.at(mt),
                          o__.ld());

        la::wrap(la).gemm('N', 'T', kp_.num_gkvec_row(), kp_.num_gkvec_col(), num_mt_aw,
                          &la::constant<std::complex<T>>::one(), alm_row.at(mt1, 0, 0, s), alm_row.ld(),
                          halm_col.at(mt1, 0, 0, s), halm_col.ld(), &la::constant<std::complex<T>>::one(), h__.at(mt),
                          h__.ld());
    }

    // TODO: fix the logic of matrices setup
    // problem: for magma we start on CPU, for cusoler - on GPU
    // one solution: start from gpu for magma as well
    // add starting pointer type in the Eigensolver() class

    // if (pu == device_t::GPU) {
    //     acc::copyout(h__.at(memory_t::host), h__.ld(), h__.at(memory_t::device), h__.ld(), kp.num_gkvec_row(),
    //         kp.num_gkvec_col());
    //     acc::copyout(o__.at(memory_t::host), o__.ld(), o__.at(memory_t::device), o__.ld(), kp.num_gkvec_row(),
    //         kp.num_gkvec_col());
    // }
    PROFILE_STOP("sirius::Hamiltonian_k::set_fv_h_o|zgemm");
    if (env::print_performance()) {
        auto tval = time_interval(t1);
        RTE_OUT(kp_.out(0)) << "effective zgemm performance: "
                            << 2 * 8e-9 * std::pow(kp_.num_gkvec(), 2) * uc.mt_aw_basis_size() / tval << " GFlop/s"
                            << std::endl;
    }

    /* add interstitial contributon */
    set_fv_h_o_it(h__, o__);

    /* setup lo-lo block */
    set_fv_h_o_lo_lo(h__, o__);

    ///*  copy back to GPU */ // TODO: optimize the copies
    // if (pu == device_t::GPU) {
    //     acc::copyin(h__.at(memory_t::device), h__.ld(), h__.at(memory_t::host), h__.ld(), kp.gklo_basis_size_row(),
    //         kp.gklo_basis_size_col());
    //     acc::copyin(o__.at(memory_t::device), o__.ld(), o__.at(memory_t::host), o__.ld(), kp.gklo_basis_size_row(),
    //         kp.gklo_basis_size_col());
    // }
}

/* alm_row comes in already conjugated */
template <typename T>
void
Hamiltonian_k<T>::set_fv_h_o_apw_lo(Atom const& atom__, int ia__, mdarray<std::complex<T>, 2>& alm_row__,
                                    mdarray<std::complex<T>, 2>& alm_col__, mdarray<std::complex<T>, 2>& h__,
                                    mdarray<std::complex<T>, 2>& o__) const
{
    auto& type = atom__.type();
    /* apw-lo block */
    for (int i = 0; i < kp_.num_atom_lo_cols(ia__); i++) {
        int icol = kp_.lo_col(ia__, i);
        /* local orbital indices */
        int l     = kp_.lo_basis_descriptor_col(icol).l;
        int lm    = kp_.lo_basis_descriptor_col(icol).lm;
        int idxrf = kp_.lo_basis_descriptor_col(icol).idxrf;
        int order = kp_.lo_basis_descriptor_col(icol).order;
        /* loop over apw components and update H */
        for (int j1 = 0; j1 < type.mt_aw_basis_size(); j1++) {
            int lm1    = type.indexb(j1).lm;
            int idxrf1 = type.indexb(j1).idxrf;

            auto zsum = atom__.radial_integrals_sum_L3(spin_block_t::nm, idxrf, idxrf1,
                                                       type.gaunt_coefs().gaunt_vector(lm1, lm));

            if (std::abs(zsum) > 1e-14) {
                for (int igkloc = 0; igkloc < kp_.num_gkvec_row(); igkloc++) {
                    h__(igkloc, kp_.num_gkvec_col() + icol) +=
                            static_cast<std::complex<T>>(zsum) * alm_row__(igkloc, j1);
                }
            }
        }
        /* update O */
        for (int order1 = 0; order1 < type.aw_order(l); order1++) {
            int xi1 = type.indexb().index_by_lm_order(lm, order1);
            T ori   = atom__.symmetry_class().o_radial_integral(l, order1, order);
            if (H0_.ctx().valence_relativity() == relativity_t::iora) {
                auto idxrf1 = type.indexr().index_of(angular_momentum(l), order1);
                ori += atom__.symmetry_class().o1_radial_integral(idxrf1, idxrf);
            }

            for (int igkloc = 0; igkloc < kp_.num_gkvec_row(); igkloc++) {
                o__(igkloc, kp_.num_gkvec_col() + icol) += ori * alm_row__(igkloc, xi1);
            }
        }
    }

    std::vector<std::complex<T>> ztmp(kp_.num_gkvec_col());
    /* lo-apw block */
    for (int i = 0; i < kp_.num_atom_lo_rows(ia__); i++) {
        int irow = kp_.lo_row(ia__, i);
        /* local orbital indices */
        int l     = kp_.lo_basis_descriptor_row(irow).l;
        int lm    = kp_.lo_basis_descriptor_row(irow).lm;
        int idxrf = kp_.lo_basis_descriptor_row(irow).idxrf;
        int order = kp_.lo_basis_descriptor_row(irow).order;

        std::fill(ztmp.begin(), ztmp.end(), 0);

        /* loop over apw components */
        for (int j1 = 0; j1 < type.mt_aw_basis_size(); j1++) {
            int lm1    = type.indexb(j1).lm;
            int idxrf1 = type.indexb(j1).idxrf;

            auto zsum = atom__.radial_integrals_sum_L3(spin_block_t::nm, idxrf1, idxrf,
                                                       type.gaunt_coefs().gaunt_vector(lm, lm1));

            if (std::abs(zsum) > 1e-14) {
                for (int igkloc = 0; igkloc < kp_.num_gkvec_col(); igkloc++) {
                    ztmp[igkloc] += static_cast<std::complex<T>>(zsum) * alm_col__(igkloc, j1);
                }
            }
        }

        for (int igkloc = 0; igkloc < kp_.num_gkvec_col(); igkloc++) {
            h__(irow + kp_.num_gkvec_row(), igkloc) += ztmp[igkloc];
        }

        for (int order1 = 0; order1 < type.aw_order(l); order1++) {
            int xi1 = type.indexb().index_by_lm_order(lm, order1);
            T ori   = atom__.symmetry_class().o_radial_integral(l, order, order1);
            if (H0_.ctx().valence_relativity() == relativity_t::iora) {
                int idxrf1 = type.indexr().index_of(angular_momentum(l), order1);
                ori += atom__.symmetry_class().o1_radial_integral(idxrf, idxrf1);
            }

            for (int igkloc = 0; igkloc < kp_.num_gkvec_col(); igkloc++) {
                o__(irow + kp_.num_gkvec_row(), igkloc) += ori * alm_col__(igkloc, xi1);
            }
        }
    }
}

template <typename T>
void
Hamiltonian_k<T>::set_fv_h_o_lo_lo(la::dmatrix<std::complex<T>>& h__, la::dmatrix<std::complex<T>>& o__) const
{
    PROFILE("sirius::Hamiltonian_k::set_fv_h_o_lo_lo");

    auto& kp = this->kp_;

    /* lo-lo block */
    #pragma omp parallel for default(shared)
    for (int icol = 0; icol < kp.num_lo_col(); icol++) {
        int ia     = kp.lo_basis_descriptor_col(icol).ia;
        int lm2    = kp.lo_basis_descriptor_col(icol).lm;
        int idxrf2 = kp.lo_basis_descriptor_col(icol).idxrf;

        for (int irow = 0; irow < kp.num_lo_row(); irow++) {
            /* lo-lo block is diagonal in atom index */
            if (ia == kp.lo_basis_descriptor_row(irow).ia) {
                auto& atom = H0_.ctx().unit_cell().atom(ia);
                int lm1    = kp.lo_basis_descriptor_row(irow).lm;
                int idxrf1 = kp.lo_basis_descriptor_row(irow).idxrf;

                h__(kp.num_gkvec_row() + irow, kp.num_gkvec_col() + icol) += atom.radial_integrals_sum_L3(
                        spin_block_t::nm, idxrf1, idxrf2, atom.type().gaunt_coefs().gaunt_vector(lm1, lm2));

                if (lm1 == lm2) {
                    int l      = kp.lo_basis_descriptor_row(irow).l;
                    int order1 = kp.lo_basis_descriptor_row(irow).order;
                    int order2 = kp.lo_basis_descriptor_col(icol).order;
                    o__(kp.num_gkvec_row() + irow, kp.num_gkvec_col() + icol) +=
                            atom.symmetry_class().o_radial_integral(l, order1, order2);
                    if (H0_.ctx().valence_relativity() == relativity_t::iora) {
                        auto idxrf1 = atom.type().indexr().index_of(angular_momentum(l), order1);
                        auto idxrf2 = atom.type().indexr().index_of(angular_momentum(l), order2);
                        o__(kp.num_gkvec_row() + irow, kp.num_gkvec_col() + icol) +=
                                atom.symmetry_class().o1_radial_integral(idxrf1, idxrf2);
                    }
                }
            }
        }
    }
}

template <typename T>
void
Hamiltonian_k<T>::set_fv_h_o_it(la::dmatrix<std::complex<T>>& h__, la::dmatrix<std::complex<T>>& o__) const
{
    PROFILE("sirius::Hamiltonian_k::set_fv_h_o_it");

    double sq_alpha_half = 0.5 * std::pow(speed_of_light, -2);

    auto& kp = this->kp_;

    #pragma omp parallel for default(shared)
    for (int igk_col = 0; igk_col < kp.num_gkvec_col(); igk_col++) {
        /* fractional coordinates of G vectors */
        auto gvec_col = kp.gkvec_col().gvec(gvec_index_t::local(igk_col));
        /* Cartesian coordinates of G+k vectors */
        auto gkvec_col_cart = kp.gkvec_col().gkvec_cart(gvec_index_t::local(igk_col));
        for (int igk_row = 0; igk_row < kp.num_gkvec_row(); igk_row++) {
            auto gvec_row       = kp.gkvec_row().gvec(gvec_index_t::local(igk_row));
            auto gkvec_row_cart = kp.gkvec_row().gkvec_cart(gvec_index_t::local(igk_row));
            int ig12            = H0().ctx().gvec().index_g12(gvec_row, gvec_col);
            /* pw kinetic energy */
            double t1 = 0.5 * r3::dot(gkvec_row_cart, gkvec_col_cart);

            h__(igk_row, igk_col) += H0().potential().veff_pw(ig12);
            o__(igk_row, igk_col) += H0().ctx().theta_pw(ig12);

            switch (H0().ctx().valence_relativity()) {
                case relativity_t::iora: {
                    h__(igk_row, igk_col) += t1 * H0().potential().rm_inv_pw(ig12);
                    o__(igk_row, igk_col) += t1 * sq_alpha_half * H0().potential().rm2_inv_pw(ig12);
                    break;
                }
                case relativity_t::zora: {
                    h__(igk_row, igk_col) += t1 * H0().potential().rm_inv_pw(ig12);
                    break;
                }
                default: {
                    h__(igk_row, igk_col) += t1 * H0().ctx().theta_pw(ig12);
                }
            }
        }
    }
}

//== template <spin_block_t sblock>
//== void Band::apply_uj_correction(mdarray<std::complex<double>, 2>& fv_states, mdarray<std::complex<double>, 3>& hpsi)
//== {
//==     Timer t("sirius::Band::apply_uj_correction");
//==
//==     for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//==     {
//==         if (unit_cell_.atom(ia)->apply_uj_correction())
//==         {
//==             Atom_type* type = unit_cell_.atom(ia)->type();
//==
//==             int offset = unit_cell_.atom(ia)->offset_wf();
//==
//==             int l = unit_cell_.atom(ia)->uj_correction_l();
//==
//==             int nrf = type->indexr().num_rf(l);
//==
//==             for (int order2 = 0; order2 < nrf; order2++)
//==             {
//==                 for (int lm2 = Utils::lm_by_l_m(l, -l); lm2 <= Utils::lm_by_l_m(l, l); lm2++)
//==                 {
//==                     int idx2 = type->indexb_by_lm_order(lm2, order2);
//==                     for (int order1 = 0; order1 < nrf; order1++)
//==                     {
//==                         double ori = unit_cell_.atom(ia)->symmetry_class()->o_radial_integral(l, order2, order1);
//==
//==                         for (int ist = 0; ist < parameters_.spl_fv_states().local_size(); ist++)
//==                         {
//==                             for (int lm1 = Utils::lm_by_l_m(l, -l); lm1 <= Utils::lm_by_l_m(l, l); lm1++)
//==                             {
//==                                 int idx1 = type->indexb_by_lm_order(lm1, order1);
//==                                 std::complex<double> z1 = fv_states(offset + idx1, ist) * ori;
//==
//==                                 if (sblock == uu)
//==                                 {
//==                                     hpsi(offset + idx2, ist, 0) += z1 *
//==                                         unit_cell_.atom(ia)->uj_correction_matrix(lm2, lm1, 0, 0);
//==                                 }
//==
//==                                 if (sblock == dd)
//==                                 {
//==                                     hpsi(offset + idx2, ist, 1) += z1 *
//==                                         unit_cell_.atom(ia)->uj_correction_matrix(lm2, lm1, 1, 1);
//==                                 }
//==
//==                                 if (sblock == ud)
//==                                 {
//==                                     hpsi(offset + idx2, ist, 2) += z1 *
//==                                         unit_cell_.atom(ia)->uj_correction_matrix(lm2, lm1, 0, 1);
//==                                 }
//==
//==                                 if (sblock == du)
//==                                 {
//==                                     hpsi(offset + idx2, ist, 3) += z1 *
//==                                         unit_cell_.atom(ia)->uj_correction_matrix(lm2, lm1, 1, 0);
//==                                 }
//==                             }
//==                         }
//==                     }
//==                 }
//==             }
//==         }
//==     }
//== }

template <typename T>
void
Hamiltonian_k<T>::apply_fv_h_o(bool apw_only__, bool phi_is_lo__, wf::band_range b__, wf::Wave_functions<T>& phi__,
                               wf::Wave_functions<T>* hphi__, wf::Wave_functions<T>* ophi__) const
{
    PROFILE("sirius::Hamiltonian_k::apply_fv_h_o");

    /* trivial case */
    if (hphi__ == nullptr && ophi__ == nullptr) {
        return;
    }

    using Tc = std::complex<T>;

    auto& ctx = this->H0_.ctx();

    auto pu = ctx.processing_unit();

    auto la  = (pu == device_t::CPU) ? la::lib_t::blas : la::lib_t::gpublas;
    auto mem = (pu == device_t::CPU) ? memory_t::host : memory_t::device;

    auto pp  = env::print_performance();
    auto pcs = env::print_checksum();

    /* prefactor for the matrix multiplication in complex or double arithmetic (in Giga-operations) */
    double ngop{8e-9};                          // default value for complex type
    if (std::is_same<T, real_type<T>>::value) { // change it if it is real type
        ngop = 2e-9;
    }
    double gflops{0};
    double time{0};

    if (hphi__ != nullptr) {
        hphi__->zero(mem, wf::spin_index(0), b__);
    }
    if (ophi__ != nullptr) {
        ophi__->zero(mem, wf::spin_index(0), b__);
        /* in case of GPU muffin-tin part of ophi is computed on the CPU */
        if (is_device_memory(mem)) {
            ophi__->zero(memory_t::host, wf::spin_index(0), b__);
        }
    }

    auto& comm = kp_.comm();

    /* ophi is computed on the CPU to avoid complicated GPU implementation */
    if (is_device_memory(mem)) {
        phi__.copy_mt_to(memory_t::host, wf::spin_index(0), b__);
    }

    if (pcs) {
        auto cs = phi__.checksum(mem, wf::spin_index(0), b__);
        if (comm.rank() == 0) {
            print_checksum("phi", cs, RTE_OUT(std::cout));
        }
    }

    if (!phi_is_lo__) {
        /* interstitial part */
        H0_.local_op().apply_fplapw(reinterpret_cast<fft::spfft_transform_type<T>&>(kp_.spfft_transform()),
                                    kp_.gkvec_fft_sptr(), b__, phi__, hphi__, ophi__, nullptr, nullptr);

        if (pcs) {
            if (hphi__) {
                auto cs    = hphi__->checksum(mem, wf::spin_index(0), b__);
                auto cs_pw = hphi__->checksum_pw(mem, wf::spin_index(0), b__);
                auto cs_mt = hphi__->checksum_mt(mem, wf::spin_index(0), b__);
                if (comm.rank() == 0) {
                    print_checksum("hloc_phi_pw", cs_pw, RTE_OUT(std::cout));
                    print_checksum("hloc_phi_mt", cs_mt, RTE_OUT(std::cout));
                    print_checksum("hloc_phi", cs, RTE_OUT(std::cout));
                }
            }
            if (ophi__) {
                auto cs = ophi__->checksum(mem, wf::spin_index(0), b__);
                if (comm.rank() == 0) {
                    print_checksum("oloc_phi", cs, RTE_OUT(std::cout));
                }
            }
        }
    }

    /* short name for local number of G+k vectors */
    int ngv = kp_.num_gkvec_loc();

    auto& spl_atoms = phi__.spl_num_atoms();

    /* block size of scalapack distribution */
    int bs = ctx.cyclic_block_size();

    auto& one  = la::constant<Tc>::one();
    auto& zero = la::constant<Tc>::zero();

    /* apply APW-lo part of Hamiltonian to lo- part of wave-functions */
    auto apply_hmt_apw_lo = [this, &ctx, &phi__, la, mem, &b__, &spl_atoms](wf::Wave_functions_mt<T>& h_apw_lo__) {
        #pragma omp parallel for
        for (auto it : spl_atoms) {
            int tid    = omp_get_thread_num();
            int ia     = it.i;
            auto& atom = ctx.unit_cell().atom(ia);
            auto& type = atom.type();
            int naw    = type.mt_aw_basis_size();
            int nlo    = type.mt_lo_basis_size();

            auto aidx = it.li;

            auto& hmt = this->H0_.hmt(ia);

            la::wrap(la).gemm('N', 'N', naw, b__.size(), nlo, &la::constant<Tc>::one(), hmt.at(mem, 0, naw), hmt.ld(),
                              phi__.at(mem, 0, aidx, wf::spin_index(0), wf::band_index(b__.begin())), phi__.ld(),
                              &la::constant<Tc>::zero(),
                              h_apw_lo__.at(mem, 0, aidx, wf::spin_index(0), wf::band_index(0)), h_apw_lo__.ld(),
                              acc::stream_id(tid));
        }
        if (is_device_memory(mem)) {
            h_apw_lo__.copy_to(memory_t::host);
        }
    };

    /* apply APW-lo part of overlap matrix to lo- part of wave-functions */
    auto apply_omt_apw_lo = [this, &ctx, &phi__, &b__, &spl_atoms](wf::Wave_functions_mt<T>& o_apw_lo__) {
        o_apw_lo__.zero(memory_t::host, wf::spin_index(0), wf::band_range(0, b__.size()));

        #pragma omp parallel for
        for (auto it : spl_atoms) {
            int ia     = it.i;
            auto& atom = ctx.unit_cell().atom(ia);
            auto& type = atom.type();
            int naw    = type.mt_aw_basis_size();
            int nlo    = type.mt_lo_basis_size();

            auto aidx = it.li;

            for (int j = 0; j < b__.size(); j++) {
                for (int ilo = 0; ilo < nlo; ilo++) {
                    int xi_lo = naw + ilo;
                    /* local orbital indices */
                    int l_lo     = type.indexb(xi_lo).am.l();
                    int lm_lo    = type.indexb(xi_lo).lm;
                    int order_lo = type.indexb(xi_lo).order;
                    for (int order_aw = 0; order_aw < (int)type.aw_descriptor(l_lo).size(); order_aw++) {
                        int xi = type.indexb_by_lm_order(lm_lo, order_aw);
                        o_apw_lo__.mt_coeffs(xi, aidx, wf::spin_index(0), wf::band_index(j)) +=
                                phi__.mt_coeffs(ilo, aidx, wf::spin_index(0), wf::band_index(b__.begin() + j)) *
                                static_cast<T>(atom.symmetry_class().o_radial_integral(l_lo, order_aw, order_lo));
                    }
                }
            }
        }
    };

    auto appy_hmt_lo_lo = [this, &ctx, &phi__, la, mem, &b__, &spl_atoms](wf::Wave_functions<T>& hphi__) {
        /* lo-lo contribution */
        #pragma omp parallel for
        for (auto it : spl_atoms) {
            int tid    = omp_get_thread_num();
            auto ia    = it.i;
            auto& atom = ctx.unit_cell().atom(ia);
            auto& type = atom.type();
            int naw    = type.mt_aw_basis_size();
            int nlo    = type.mt_lo_basis_size();

            auto aidx = it.li;

            auto& hmt = H0_.hmt(ia);

            la::wrap(la).gemm('N', 'N', nlo, b__.size(), nlo, &la::constant<Tc>::one(), hmt.at(mem, naw, naw), hmt.ld(),
                              phi__.at(mem, 0, aidx, wf::spin_index(0), wf::band_index(b__.begin())), phi__.ld(),
                              &la::constant<Tc>::one(),
                              hphi__.at(mem, 0, aidx, wf::spin_index(0), wf::band_index(b__.begin())), hphi__.ld(),
                              acc::stream_id(tid));
        }
    };

    auto appy_omt_lo_lo = [this, &ctx, &phi__, &b__, &spl_atoms](wf::Wave_functions<T>& ophi__) {
        /* lo-lo contribution */
        #pragma omp parallel for
        for (auto it : spl_atoms) {
            auto ia    = it.i;
            auto& atom = ctx.unit_cell().atom(ia);
            auto& type = atom.type();

            auto aidx = it.li;

            for (int ilo = 0; ilo < type.mt_lo_basis_size(); ilo++) {
                int xi_lo = type.mt_aw_basis_size() + ilo;
                /* local orbital indices */
                int l_lo     = type.indexb(xi_lo).am.l();
                int lm_lo    = type.indexb(xi_lo).lm;
                int order_lo = type.indexb(xi_lo).order;

                /* lo-lo contribution */
                for (int jlo = 0; jlo < type.mt_lo_basis_size(); jlo++) {
                    int xi_lo1 = type.mt_aw_basis_size() + jlo;
                    int lm1    = type.indexb(xi_lo1).lm;
                    int order1 = type.indexb(xi_lo1).order;
                    if (lm_lo == lm1) {
                        for (int i = 0; i < b__.size(); i++) {
                            ophi__.mt_coeffs(ilo, aidx, wf::spin_index(0), wf::band_index(b__.begin() + i)) +=
                                    phi__.mt_coeffs(jlo, aidx, wf::spin_index(0), wf::band_index(b__.begin() + i)) *
                                    static_cast<T>(atom.symmetry_class().o_radial_integral(l_lo, order_lo, order1));
                        }
                    }
                }
            }
        }
    };

    auto appy_hmt_apw_apw = [this, &ctx, la, mem, &b__](int atom_begin__, wf::Wave_functions_mt<T> const& alm_phi__,
                                                        wf::Wave_functions_mt<T>& halm_phi__) {
        #pragma omp parallel for
        for (auto it : alm_phi__.spl_num_atoms()) {
            int tid    = omp_get_thread_num();
            int ia     = atom_begin__ + it.i;
            auto& atom = ctx.unit_cell().atom(ia);
            auto& type = atom.type();
            int naw    = type.mt_aw_basis_size();

            auto aidx = it.li;

            auto& hmt = H0_.hmt(ia);

            // TODO: use in-place trmm
            la::wrap(la).gemm('N', 'N', naw, b__.size(), naw, &la::constant<Tc>::one(), hmt.at(mem), hmt.ld(),
                              alm_phi__.at(mem, 0, aidx, wf::spin_index(0), wf::band_index(0)), alm_phi__.ld(),
                              &la::constant<Tc>::zero(),
                              halm_phi__.at(mem, 0, aidx, wf::spin_index(0), wf::band_index(0)), halm_phi__.ld(),
                              acc::stream_id(tid));
        }
    };

    auto apply_hmt_lo_apw = [this, &ctx, la, mem, &b__, &spl_atoms](wf::Wave_functions_mt<T> const& alm_phi__,
                                                                    wf::Wave_functions<T>& hphi__) {
        #pragma omp parallel for
        for (auto it : spl_atoms) {
            int tid    = omp_get_thread_num();
            int ia     = it.i;
            auto& atom = ctx.unit_cell().atom(ia);
            auto& type = atom.type();
            int naw    = type.mt_aw_basis_size();
            int nlo    = type.mt_lo_basis_size();

            auto aidx = it.li;

            auto& hmt = H0_.hmt(ia);

            // TODO: add stream_id

            la::wrap(la).gemm('N', 'N', nlo, b__.size(), naw, &la::constant<Tc>::one(), hmt.at(mem, naw, 0), hmt.ld(),
                              alm_phi__.at(mem, 0, aidx, wf::spin_index(0), wf::band_index(0)), alm_phi__.ld(),
                              &la::constant<Tc>::one(),
                              hphi__.at(mem, 0, aidx, wf::spin_index(0), wf::band_index(b__.begin())), hphi__.ld(),
                              acc::stream_id(tid));
        }
    };

    auto apply_omt_lo_apw = [this, &ctx, mem, &b__, &spl_atoms](wf::Wave_functions_mt<T> const& alm_phi__,
                                                                wf::Wave_functions<T>& ophi__) {
        #pragma omp parallel for
        for (auto it : spl_atoms) {
            int ia     = it.i;
            auto& atom = ctx.unit_cell().atom(ia);
            auto& type = atom.type();
            int naw    = type.mt_aw_basis_size();
            int nlo    = type.mt_lo_basis_size();

            auto aidx = it.li;

            for (int ilo = 0; ilo < nlo; ilo++) {
                int xi_lo = naw + ilo;
                /* local orbital indices */
                int l_lo     = type.indexb(xi_lo).am.l();
                int lm_lo    = type.indexb(xi_lo).lm;
                int order_lo = type.indexb(xi_lo).order;
                for (int i = 0; i < b__.size(); i++) {
                    /* lo-APW contribution to ophi */
                    for (int order_aw = 0; order_aw < (int)type.aw_descriptor(l_lo).size(); order_aw++) {
                        ophi__.mt_coeffs(ilo, aidx, wf::spin_index(0), wf::band_index(b__.begin() + i)) +=
                                static_cast<T>(atom.symmetry_class().o_radial_integral(l_lo, order_lo, order_aw)) *
                                alm_phi__.mt_coeffs(type.indexb_by_lm_order(lm_lo, order_aw), aidx, wf::spin_index(0),
                                                    wf::band_index(i));
                    }
                }
            }
        }
    };

    PROFILE_START("sirius::Hamiltonian_k::apply_fv_h_o|mt");

    /*
     * Application of LAPW Hamiltonian splits into four parts:
     *                                                            n                    n
     *                               n     +----------------+   +---+     +------+   +---+
     * +----------------+------+   +---+   |                |   |   |     |      |   |   |
     * |                |      |   |   |   |                |   |   |     |      | x |lo |
     * |                |      |   |   |   |                |   |   |     |      |   |   |
     * |                |      |   |   |   |                |   |   |     |      |   +---+
     * |                |      |   |   |   |    APW-APW     | x |APW|  +  |APW-lo|
     * |     APW-APW    |APW-lo|   |APW|   |                |   |   |     |      |
     * |                |      |   |   |   |                |   |   |     |      |
     * |                |      | x |   |   |                |   |   |     |      |
     * |                |      |   |   |   +----------------+   +---+     +------+
     * +----------------+------+   +---+ =
     * |                |      |   |   |   +----------------+   +---+     +------+   +---+
     * |    lo-APW      |lo-lo |   |lo |   |                |   |   |     |      |   |   |
     * |                |      |   |   |   |     lo-APW     | x |   |  +  |lo-lo | x |lo |
     * +----------------+------+   +---+   |                |   |   |     |      |   |   |
     *                                     +----------------+   |   |     +------+   +---+
     *                                                          |APW|
     *                                                          |   |
     *                                                          |   |
     *                                                          |   |
     *                                                          |   |
     *                                                          +---+
     */

    /* Prepare APW-lo contribution for the entire index of APW basis functions. Here we compute the action
     * of the APW-lo Hamiltonian and overlap on the local-orbital part of wave-functions.
     *
     *            n
     * +------+ +---+
     * |      | |   |
     * |      |x|lo |
     * |      | |   |
     * |      | +---+
     * |APW-lo|
     * |      |
     * |      |
     * |      |
     * +------+
     */
    la::dmatrix<Tc> h_apw_lo_phi_lo;
    la::dmatrix<Tc> o_apw_lo_phi_lo;

    std::vector<int> num_mt_apw_coeffs(ctx.unit_cell().num_atoms());
    for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
        num_mt_apw_coeffs[ia] = ctx.unit_cell().atom(ia).mt_aw_basis_size();
    }
    wf::Wave_functions_mt<T> tmp(comm, num_mt_apw_coeffs, wf::num_mag_dims(0), wf::num_bands(b__.size()),
                                 memory_t::host);
    auto mg = tmp.memory_guard(mem);

    if (!apw_only__ && ctx.unit_cell().mt_lo_basis_size()) {
        PROFILE("sirius::Hamiltonian_k::apply_fv_h_o|apw-lo-prep");

        if (hphi__) {
            apply_hmt_apw_lo(tmp);

            h_apw_lo_phi_lo = la::dmatrix<Tc>(ctx.unit_cell().mt_aw_basis_size(), b__.size(), ctx.blacs_grid(), bs, bs);

            auto layout_in = tmp.grid_layout_mt(wf::spin_index(0), wf::band_range(0, b__.size()));
            costa::transform(layout_in, h_apw_lo_phi_lo.grid_layout(), 'N', one, zero, comm.native());
        }
        if (ophi__) {
            apply_omt_apw_lo(tmp);
            o_apw_lo_phi_lo = la::dmatrix<Tc>(ctx.unit_cell().mt_aw_basis_size(), b__.size(), ctx.blacs_grid(), bs, bs);

            auto layout_in = tmp.grid_layout_mt(wf::spin_index(0), wf::band_range(0, b__.size()));
            costa::transform(layout_in, o_apw_lo_phi_lo.grid_layout(), 'N', one, zero, comm.native());
        }
    }

    // here: h_apw_lo_phi_lo is on host
    //       o_apw_lo_phi_lo is on host

    /* lo-lo contribution (lo-lo Hamiltonian and overlap are block-diagonal in atom index and the whole application is
     * local to MPI rank)
     *
     *            n
     * +------+ +---+
     * |      | |   |
     * |lo-lo |x|lo |
     * |      | |   |
     * +------+ +---+
     */
    if (!apw_only__ && ctx.unit_cell().mt_lo_basis_size()) {
        PROFILE("sirius::Hamiltonian_k::apply_fv_h_o|lo-lo");
        if (hphi__) {
            appy_hmt_lo_lo(*hphi__);
        }
        if (ophi__) {
            appy_omt_lo_lo(*ophi__);
        }
    }

    /* <A_{lm}^{\alpha}(G) | C_j(G) > for all Alm matching coefficients */
    la::dmatrix<Tc> alm_phi(ctx.unit_cell().mt_aw_basis_size(), b__.size(), ctx.blacs_grid(), bs, bs);

    /*  compute APW-APW contribution
     *                         n
     *  +----------------+   +---+
     *  |                |   |   |
     *  |                |   |   |
     *  |                |   |   |
     *  |                |   |   |
     *  |    APW-APW     | x |APW|
     *  |                |   |   |
     *  |                |   |   |
     *  |                |   |   |
     *  +----------------+   +---+
     *
     *  we are going to split the Alm coefficients into blocks of atoms and work on them consecutively
     */
    int offset_aw_global{0};
    int atom_begin{0};
    /* loop over blocks of atoms */
    for (auto na : split_in_blocks(ctx.unit_cell().num_atoms(), 64)) {

        splindex_block<> spl_atoms(na, n_blocks(comm.size()), block_id(comm.rank()));

        /* actual number of AW radial functions in a block of atoms */
        int num_mt_aw{0};
        std::vector<int> offsets_aw(na);
        for (int i = 0; i < na; i++) {
            int ia        = atom_begin + i;
            auto& type    = ctx.unit_cell().atom(ia).type();
            offsets_aw[i] = num_mt_aw;
            num_mt_aw += type.mt_aw_basis_size();
        }

        /* generate complex conjugated Alm coefficients for a block of atoms */
        auto alm = generate_alm_block<true, T>(ctx, atom_begin, na, kp_.alm_coeffs_loc());

        /* if there is APW part */
        if (!phi_is_lo__) {
            auto t0 = time_now();

            PROFILE("sirius::Hamiltonian_k::apply_fv_h_o|apw-apw");

            /* compute alm_phi(lm, n) = < Alm | C > */
            spla::pgemm_ssb(num_mt_aw, b__.size(), ngv, SPLA_OP_CONJ_TRANSPOSE, 1.0, alm.at(mem), alm.ld(),
                            phi__.at(mem, 0, wf::spin_index(0), wf::band_index(b__.begin())), phi__.ld(), 0.0,
                            alm_phi.at(memory_t::host), alm_phi.ld(), offset_aw_global, 0, alm_phi.spla_distribution(),
                            ctx.spla_context());
            gflops += ngop * num_mt_aw * b__.size() * ngv;

            if (pcs) {
                auto cs = alm_phi.checksum(num_mt_aw, b__.size());
                if (comm.rank() == 0) {
                    print_checksum("alm_phi", cs, RTE_OUT(std::cout));
                }
            }

            if (ophi__) {
                /* add APW-APW contribution to ophi */
                spla::pgemm_sbs(ngv, b__.size(), num_mt_aw, one, alm.at(mem), alm.ld(), alm_phi.at(memory_t::host),
                                alm_phi.ld(), offset_aw_global, 0, alm_phi.spla_distribution(), one,
                                ophi__->at(mem, 0, wf::spin_index(0), wf::band_index(b__.begin())), ophi__->ld(),
                                ctx.spla_context());
                gflops += ngop * ngv * b__.size() * num_mt_aw;
            }

            if (hphi__) {
                std::vector<int> num_mt_apw_coeffs_in_block(na);
                for (int i = 0; i < na; i++) {
                    num_mt_apw_coeffs_in_block[i] = ctx.unit_cell().atom(atom_begin + i).mt_aw_basis_size();
                }

                wf::Wave_functions_mt<T> alm_phi_slab(comm, num_mt_apw_coeffs_in_block, wf::num_mag_dims(0),
                                                      wf::num_bands(b__.size()), memory_t::host);
                wf::Wave_functions_mt<T> halm_phi_slab(comm, num_mt_apw_coeffs_in_block, wf::num_mag_dims(0),
                                                       wf::num_bands(b__.size()), memory_t::host);

                la::dmatrix<Tc> halm_phi(num_mt_aw, b__.size(), ctx.blacs_grid(), bs, bs);
                {
                    auto layout_in  = alm_phi.grid_layout(offset_aw_global, 0, num_mt_aw, b__.size());
                    auto layout_out = alm_phi_slab.grid_layout_mt(wf::spin_index(0), wf::band_range(0, b__.size()));
                    costa::transform(layout_in, layout_out, 'N', one, zero, comm.native());
                }

                {
                    auto mg1 = alm_phi_slab.memory_guard(mem, wf::copy_to::device);
                    auto mg2 = halm_phi_slab.memory_guard(mem, wf::copy_to::host);
                    appy_hmt_apw_apw(atom_begin, alm_phi_slab, halm_phi_slab);

                    if (pcs) {
                        auto cs1 = alm_phi_slab.checksum_mt(memory_t::host, wf::spin_index(0), b__);
                        auto cs2 = halm_phi_slab.checksum_mt(memory_t::host, wf::spin_index(0), b__);
                        if (comm.rank() == 0) {
                            print_checksum("alm_phi_slab", cs1, RTE_OUT(std::cout));
                            print_checksum("halm_phi_slab", cs2, RTE_OUT(std::cout));
                        }
                    }
                }

                {
                    auto layout_in  = halm_phi_slab.grid_layout_mt(wf::spin_index(0), wf::band_range(0, b__.size()));
                    auto layout_out = halm_phi.grid_layout();
                    costa::transform(layout_in, layout_out, 'N', one, zero, comm.native());
                    if (pcs) {
                        auto cs = halm_phi.checksum(num_mt_aw, b__.size());
                        if (comm.rank() == 0) {
                            print_checksum("halm_phi", cs, RTE_OUT(std::cout));
                        }
                    }
                }

                /* APW-APW contribution to hphi */
                spla::pgemm_sbs(ngv, b__.size(), num_mt_aw, one, alm.at(mem), alm.ld(), halm_phi.at(memory_t::host),
                                halm_phi.ld(), 0, 0, halm_phi.spla_distribution(), one,
                                hphi__->at(mem, 0, wf::spin_index(0), wf::band_index(b__.begin())), hphi__->ld(),
                                ctx.spla_context());
                gflops += ngop * ngv * b__.size() * num_mt_aw;
                if (pcs) {
                    auto cs = hphi__->checksum_pw(mem, wf::spin_index(0), b__);
                    if (comm.rank() == 0) {
                        print_checksum("hphi_apw#1", cs, RTE_OUT(std::cout));
                    }
                }
            }
            time += time_interval(t0);
        }

        if (!apw_only__ && ctx.unit_cell().mt_lo_basis_size()) {
            PROFILE("sirius::Hamiltonian_k::apply_fv_h_o|apw-lo");
            auto t0 = time_now();
            if (hphi__) {
                /* APW-lo contribution to hphi */
                spla::pgemm_sbs(ngv, b__.size(), num_mt_aw, one, alm.at(mem), alm.ld(),
                                h_apw_lo_phi_lo.at(memory_t::host), h_apw_lo_phi_lo.ld(), offset_aw_global, 0,
                                h_apw_lo_phi_lo.spla_distribution(), one,
                                hphi__->at(mem, 0, wf::spin_index(0), wf::band_index(b__.begin())), hphi__->ld(),
                                ctx.spla_context());
                if (pcs) {
                    auto cs = hphi__->checksum_pw(mem, wf::spin_index(0), b__);
                    if (comm.rank() == 0) {
                        print_checksum("hphi_apw#2", cs, RTE_OUT(std::cout));
                    }
                }
                gflops += ngop * ngv * b__.size() * num_mt_aw;
            }
            if (ophi__) {
                /* APW-lo contribution to ophi */
                spla::pgemm_sbs(ngv, b__.size(), num_mt_aw, one, alm.at(mem), alm.ld(),
                                o_apw_lo_phi_lo.at(memory_t::host), o_apw_lo_phi_lo.ld(), offset_aw_global, 0,
                                o_apw_lo_phi_lo.spla_distribution(), one,
                                ophi__->at(mem, 0, wf::spin_index(0), wf::band_index(b__.begin())), ophi__->ld(),
                                ctx.spla_context());
                gflops += ngop * ngv * b__.size() * num_mt_aw;
            }
            time += time_interval(t0);
        }
        offset_aw_global += num_mt_aw;
        atom_begin += na;

        if (pcs) {
            if (hphi__) {
                auto cs = hphi__->checksum_pw(mem, wf::spin_index(0), b__);
                if (comm.rank() == 0) {
                    print_checksum("hphi_apw#3", cs, RTE_OUT(std::cout));
                }
            }
            if (ophi__) {
                auto cs = ophi__->checksum_pw(mem, wf::spin_index(0), b__);
                if (comm.rank() == 0) {
                    print_checksum("ophi_apw", cs, RTE_OUT(std::cout));
                }
            }
        }
    } // blocks of atoms

    /* compute lo-APW contribution
     *
     *                         n
     *  +----------------+   +---+
     *  |                |   |   |
     *  |     lo-APW     | x |   |
     *  |                |   |   |
     *  +----------------+   |   |
     *                       |APW|
     *                       |   |
     *                       |   |
     *                       |   |
     *                       |   |
     *                       +---+
     */
    if (!apw_only__ && !phi_is_lo__ && ctx.unit_cell().mt_lo_basis_size()) {
        PROFILE("sirius::Hamiltonian_k::apply_fv_h_o|lo-apw");
        {
            auto layout_in  = alm_phi.grid_layout();
            auto layout_out = tmp.grid_layout_mt(wf::spin_index(0), wf::band_range(0, b__.size()));
            costa::transform(layout_in, layout_out, 'N', one, zero, comm.native());
        }

        if (ophi__) {
            apply_omt_lo_apw(tmp, *ophi__);
        }

        if (hphi__) {
            if (is_device_memory(mem)) {
                tmp.copy_to(mem);
            }
            apply_hmt_lo_apw(tmp, *hphi__);
        }
    }
    PROFILE_STOP("sirius::Hamiltonian_k::apply_fv_h_o|mt");

    if (is_device_memory(mem) && ophi__) {
        ophi__->copy_mt_to(mem, wf::spin_index(0), b__);
    }

    if (pp && comm.rank() == 0) {
        RTE_OUT(std::cout) << "effective local zgemm performance : " << gflops / time << " GFlop/s" << std::endl;
    }

    if (pcs) {
        if (hphi__) {
            auto cs = hphi__->checksum(mem, wf::spin_index(0), b__);
            if (comm.rank() == 0) {
                print_checksum("hphi", cs, RTE_OUT(std::cout));
            }
        }
        if (ophi__) {
            auto cs = ophi__->checksum(mem, wf::spin_index(0), b__);
            if (comm.rank() == 0) {
                print_checksum("ophi", cs, RTE_OUT(std::cout));
            }
        }
    }
}

template <typename T>
void
Hamiltonian_k<T>::apply_b(wf::Wave_functions<T>& psi__, std::vector<wf::Wave_functions<T>>& bpsi__) const
{
    PROFILE("sirius::Hamiltonian_k::apply_b");

    RTE_ASSERT(bpsi__.size() == 2 || bpsi__.size() == 3);

    int nfv = H0().ctx().num_fv_states();

    auto bxypsi = bpsi__.size() == 2 ? nullptr : &bpsi__[2];
    H0().local_op().apply_fplapw(reinterpret_cast<fft::spfft_transform_type<T>&>(kp_.spfft_transform()),
                                 this->kp_.gkvec_fft_sptr(), wf::band_range(0, nfv), psi__, nullptr, nullptr,
                                 &bpsi__[0], bxypsi);
    H0().apply_bmt(psi__, bpsi__);

    std::vector<T> alpha(nfv, -1.0);
    std::vector<T> beta(nfv, 0.0);

    /* copy Bz|\psi> to -Bz|\psi> */
    wf::axpby(memory_t::host, wf::spin_range(0), wf::band_range(0, nfv), alpha.data(), &bpsi__[0], beta.data(),
              &bpsi__[1]);

    auto pcs = env::print_checksum();
    if (pcs) {
        auto cs1 = bpsi__[0].checksum_pw(memory_t::host, wf::spin_index(0), wf::band_range(0, nfv));
        auto cs2 = bpsi__[0].checksum_mt(memory_t::host, wf::spin_index(0), wf::band_range(0, nfv));
        auto cs3 = bpsi__[1].checksum_pw(memory_t::host, wf::spin_index(0), wf::band_range(0, nfv));
        auto cs4 = bpsi__[1].checksum_mt(memory_t::host, wf::spin_index(0), wf::band_range(0, nfv));
        if (this->kp_.gkvec().comm().rank() == 0) {
            print_checksum("hpsi[0]_pw", cs1, RTE_OUT(std::cout));
            print_checksum("hpsi[0]_mt", cs2, RTE_OUT(std::cout));
            print_checksum("hpsi[1]_pw", cs3, RTE_OUT(std::cout));
            print_checksum("hpsi[1]_mt", cs4, RTE_OUT(std::cout));
        }
    }
}

template class Hamiltonian_k<double>;

template std::pair<mdarray<double, 2>, mdarray<double, 2>>
Hamiltonian_k<double>::get_h_o_diag_pw<double, 1>() const;

template std::pair<mdarray<double, 2>, mdarray<double, 2>>
Hamiltonian_k<double>::get_h_o_diag_pw<double, 2>() const;

template std::pair<mdarray<double, 2>, mdarray<double, 2>>
Hamiltonian_k<double>::get_h_o_diag_pw<double, 3>() const;

template std::pair<mdarray<double, 2>, mdarray<double, 2>>
Hamiltonian_k<double>::get_h_o_diag_pw<std::complex<double>, 1>() const;

template std::pair<mdarray<double, 2>, mdarray<double, 2>>
Hamiltonian_k<double>::get_h_o_diag_pw<std::complex<double>, 2>() const;

template std::pair<mdarray<double, 2>, mdarray<double, 2>>
Hamiltonian_k<double>::get_h_o_diag_pw<std::complex<double>, 3>() const;

template std::pair<mdarray<double, 2>, mdarray<double, 2>>
Hamiltonian_k<double>::get_h_o_diag_lapw<1>() const;

template std::pair<mdarray<double, 2>, mdarray<double, 2>>
Hamiltonian_k<double>::get_h_o_diag_lapw<2>() const;

template std::pair<mdarray<double, 2>, mdarray<double, 2>>
Hamiltonian_k<double>::get_h_o_diag_lapw<3>() const;

#ifdef SIRIUS_USE_FP32
template class Hamiltonian_k<float>;

template std::pair<mdarray<float, 2>, mdarray<float, 2>>
Hamiltonian_k<float>::get_h_o_diag_pw<float, 1>() const;

template std::pair<mdarray<float, 2>, mdarray<float, 2>>
Hamiltonian_k<float>::get_h_o_diag_pw<float, 2>() const;

template std::pair<mdarray<float, 2>, mdarray<float, 2>>
Hamiltonian_k<float>::get_h_o_diag_pw<float, 3>() const;

template std::pair<mdarray<float, 2>, mdarray<float, 2>>
Hamiltonian_k<float>::get_h_o_diag_pw<std::complex<float>, 1>() const;

template std::pair<mdarray<float, 2>, mdarray<float, 2>>
Hamiltonian_k<float>::get_h_o_diag_pw<std::complex<float>, 2>() const;

template std::pair<mdarray<float, 2>, mdarray<float, 2>>
Hamiltonian_k<float>::get_h_o_diag_pw<std::complex<float>, 3>() const;

template std::pair<mdarray<float, 2>, mdarray<float, 2>>
Hamiltonian_k<float>::get_h_o_diag_lapw<1>() const;

template std::pair<mdarray<float, 2>, mdarray<float, 2>>
Hamiltonian_k<float>::get_h_o_diag_lapw<2>() const;

template std::pair<mdarray<float, 2>, mdarray<float, 2>>
Hamiltonian_k<float>::get_h_o_diag_lapw<3>() const;
#endif
} // namespace sirius
