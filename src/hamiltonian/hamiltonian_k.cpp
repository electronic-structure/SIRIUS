// Copyright (c) 2013-2018 Anton Kozhevnikov, Mathieu Taillefumier, Thomas Schulthess
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

/** \file hamiltonian_k.cpp
 *
 *  \brief Contains definition of sirius::Hamiltonian_k class.
 */

#include "context/simulation_context.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "hamiltonian/local_operator.hpp"
#include "hamiltonian/non_local_operator.hpp"
#include "potential/potential.hpp"
#include "SDDK/wave_functions.hpp"
#include "SDDK/omp.hpp"
#include "k_point/k_point.hpp"
#include "utils/profiler.hpp"
#include <chrono>

namespace sirius {

template <typename T>
Hamiltonian_k<T>::Hamiltonian_k(Hamiltonian0<T>& H0__,
                                K_point<T>& kp__) // TODO: move kinetic part from local_op to here
    : H0_(H0__)
    , kp_(kp__)
{
    PROFILE("sirius::Hamiltonian_k");
    H0_.local_op().prepare_k(kp_.gkvec_partition());
    if (!H0_.ctx().full_potential()) {
        if (H0_.ctx().cfg().iterative_solver().type() != "exact") {
            kp_.beta_projectors().prepare();
        }
        u_op_ = std::shared_ptr<U_operator<T>>(
            new U_operator<T>(H0__.ctx(), H0__.potential().hubbard_potential(), kp__.vk()));
    }
}

template <typename T>
Hamiltonian_k<T>::~Hamiltonian_k()
{
    if (!H0_.ctx().full_potential()) {
        if (H0_.ctx().cfg().iterative_solver().type() != "exact") {
            kp_.beta_projectors().dismiss();
        }
    }
}

template <typename T>
Hamiltonian_k<T>::Hamiltonian_k(Hamiltonian_k&& src__) = default;

template <typename T>
template <typename F, int what>
std::pair<sddk::mdarray<T, 2>, sddk::mdarray<T, 2>>
Hamiltonian_k<T>::get_h_o_diag_pw() const
{
    PROFILE("sirius::Hamiltonian_k::get_h_o_diag");

    auto const& uc = H0_.ctx().unit_cell();

    sddk::mdarray<T, 2> h_diag(kp_.num_gkvec_loc(), H0_.ctx().num_spins());
    sddk::mdarray<T, 2> o_diag(kp_.num_gkvec_loc(), H0_.ctx().num_spins());

    h_diag.zero();
    o_diag.zero();

    for (int ispn = 0; ispn < H0_.ctx().num_spins(); ispn++) {

        /* local H contribution */
        #pragma omp parallel for schedule(static)
        for (int ig_loc = 0; ig_loc < kp_.num_gkvec_loc(); ig_loc++) {
            if (what & 1) {
                auto ekin            = 0.5 * kp_.gkvec().template gkvec_cart<index_domain_t::local>(ig_loc).length2();
                h_diag(ig_loc, ispn) = ekin + H0_.local_op().v0(ispn);
            }
            if (what & 2) {
                o_diag(ig_loc, ispn) = 1;
            }
        }
        if (uc.mt_lo_basis_size() == 0) {
            continue;
        }

        /* non-local H contribution */
        auto beta_gk_t = kp_.beta_projectors().pw_coeffs_t(0);
        matrix<std::complex<T>> beta_gk_tmp(kp_.num_gkvec_loc(), uc.max_mt_basis_size());

        for (int iat = 0; iat < uc.num_atom_types(); iat++) {
            auto& atom_type = uc.atom_type(iat);
            int nbf         = atom_type.mt_basis_size();
            if (!nbf) {
                continue;
            }

            matrix<std::complex<T>> d_sum;
            if (what & 1) {
                d_sum = matrix<std::complex<T>>(nbf, nbf);
                d_sum.zero();
            }

            matrix<std::complex<T>> q_sum;
            if (what & 2) {
                q_sum = matrix<std::complex<T>>(nbf, nbf);
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

            int offs = uc.atom_type(iat).offset_lo();

            if (what & 1) {
                sddk::linalg(linalg_t::blas)
                    .gemm('N', 'N', kp_.num_gkvec_loc(), nbf, nbf, &sddk::linalg_const<std::complex<T>>::one(),
                          &beta_gk_t(0, offs), beta_gk_t.ld(), &d_sum(0, 0), d_sum.ld(),
                          &sddk::linalg_const<std::complex<T>>::zero(), &beta_gk_tmp(0, 0), beta_gk_tmp.ld());
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
                sddk::linalg(linalg_t::blas)
                    .gemm('N', 'N', kp_.num_gkvec_loc(), nbf, nbf, &sddk::linalg_const<std::complex<T>>::one(),
                          &beta_gk_t(0, offs), beta_gk_t.ld(), &q_sum(0, 0), q_sum.ld(),
                          &sddk::linalg_const<std::complex<T>>::zero(), &beta_gk_tmp(0, 0), beta_gk_tmp.ld());
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

    splindex<splindex_t::block> spl_num_atoms(uc.num_atoms(), kp_.comm().size(), kp_.comm().rank());
    int nlo{0};
    for (int ialoc = 0; ialoc < spl_num_atoms.local_size(); ialoc++) {
        int ia = spl_num_atoms[ialoc];
        nlo += uc.atom(ia).mt_lo_basis_size();
    }

    mdarray<T, 2> h_diag = (what & 1) ? mdarray<T, 2>(kp_.num_gkvec_loc() + nlo, 1) : mdarray<T, 2>();
    mdarray<T, 2> o_diag = (what & 2) ? mdarray<T, 2>(kp_.num_gkvec_loc() + nlo, 1) : mdarray<T, 2>();

    #pragma omp parallel for schedule(static)
    for (int igloc = 0; igloc < kp_.num_gkvec_loc(); igloc++) {
        if (what & 1) {
            auto gvc      = kp_.gkvec().template gkvec_cart<index_domain_t::local>(igloc);
            T ekin        = 0.5 * dot(gvc, gvc);
            h_diag[igloc] = H0_.local_op().v0(0) + ekin * H0_.ctx().theta_pw(0).real();
        }
        if (what & 2) {
            o_diag[igloc] = H0_.ctx().theta_pw(0).real();
        }
    }

    #pragma omp parallel
    {
        matrix<std::complex<T>> alm(kp_.num_gkvec_loc(), uc.max_mt_aw_basis_size());

        matrix<std::complex<T>> halm = (what & 1)
                                           ? matrix<std::complex<T>>(kp_.num_gkvec_loc(), uc.max_mt_aw_basis_size())
                                           : matrix<std::complex<T>>();

        auto h_diag_omp = (what & 1) ? mdarray<T, 1>(kp_.num_gkvec_loc()) : mdarray<T, 1>();
        if (what & 1) {
            h_diag_omp.zero();
        }

        auto o_diag_omp = (what & 2) ? mdarray<T, 1>(kp_.num_gkvec_loc()) : mdarray<T, 1>();
        if (what & 2) {
            o_diag_omp.zero();
        }

        #pragma omp for
        for (int ia = 0; ia < uc.num_atoms(); ia++) {
            auto& atom = uc.atom(ia);
            int nmt    = atom.mt_aw_basis_size();

            kp_.alm_coeffs_loc().template generate<false>(atom, alm);
            if (what & 1) {
                H0_.template apply_hmt_to_apw<spin_block_t::nm>(atom, kp_.num_gkvec_loc(), alm, halm);
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
    for (int ialoc = 0; ialoc < spl_num_atoms.local_size(); ialoc++) {
        int ia     = spl_num_atoms[ialoc];
        auto& atom = uc.atom(ia);
        auto& type = atom.type();
        #pragma omp parallel for
        for (int ilo = 0; ilo < type.mt_lo_basis_size(); ilo++) {
            int xi_lo = type.mt_aw_basis_size() + ilo;
            /* local orbital indices */
            int lm_lo    = type.indexb(xi_lo).lm;
            int idxrf_lo = type.indexb(xi_lo).idxrf;

            if (what & 1) {
                h_diag[kp_.num_gkvec_loc() + nlo + ilo] =
                    atom.template radial_integrals_sum_L3<spin_block_t::nm>(
                            idxrf_lo, idxrf_lo, H0_.potential().gaunt_coefs().gaunt_vector(lm_lo, lm_lo))
                        .real();
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
Hamiltonian_k<T>::set_fv_h_o(sddk::dmatrix<std::complex<T>>& h__, sddk::dmatrix<std::complex<T>>& o__) const
{
    PROFILE("sirius::Hamiltonian_k::set_fv_h_o");

    /* alias to unit cell */
    auto& uc = H0_.ctx().unit_cell();
    /* alias to k-point */
    auto& kp = this->kp();
    /* split atoms in blocks */
    int num_atoms_in_block = 2 * omp_get_max_threads();
    int nblk               = uc.num_atoms() / num_atoms_in_block + std::min(1, uc.num_atoms() % num_atoms_in_block);
    /* maximum number of apw coefficients in the block of atoms */
    int max_mt_aw = num_atoms_in_block * uc.max_mt_aw_basis_size();
    /* current processing unit */
    auto pu = H0_.ctx().processing_unit();

    auto la  = linalg_t::none;
    auto mt  = memory_t::none;
    auto mt1 = memory_t::none;
    int nb   = 0;
    switch (pu) {
        case device_t::CPU: {
            la  = linalg_t::blas;
            mt  = memory_t::host;
            mt1 = memory_t::host;
            nb  = 1;
            break;
        }
        case device_t::GPU: {
            la  = linalg_t::spla;
            mt  = memory_t::host_pinned;
            mt1 = memory_t::device;
            nb  = 1;
            break;
        }
    }

    sddk::mdarray<std::complex<T>, 3> alm_row(kp.num_gkvec_row(), max_mt_aw, nb, H0_.ctx().mem_pool(mt));
    sddk::mdarray<std::complex<T>, 3> alm_col(kp.num_gkvec_col(), max_mt_aw, nb, H0_.ctx().mem_pool(mt));
    sddk::mdarray<std::complex<T>, 3> halm_col(kp.num_gkvec_col(), max_mt_aw, nb, H0_.ctx().mem_pool(mt));

    H0_.ctx().print_memory_usage(__FILE__, __LINE__);

    h__.zero();
    o__.zero();
    switch (pu) {
        case device_t::GPU: {
            //        alm_row = mdarray<std::complex<T>, 3>(kp.num_gkvec_row(), max_mt_aw, 2,
            //        H0_.ctx().mem_pool(memory_t::host_pinned)); alm_col = mdarray<std::complex<T>,
            //        3>(kp.num_gkvec_col(), max_mt_aw, 2, H0_.ctx().mem_pool(memory_t::host_pinned)); halm_col =
            //        mdarray<std::complex<T>, 3>(kp.num_gkvec_col(), max_mt_aw, 2,
            //        H0_.ctx().mem_pool(memory_t::host_pinned));
            alm_row.allocate(H0_.ctx().mem_pool(memory_t::device));
            alm_col.allocate(H0_.ctx().mem_pool(memory_t::device));
            halm_col.allocate(H0_.ctx().mem_pool(memory_t::device));
            //        h__.zero(memory_t::device);
            //        o__.zero(memory_t::device);
            break;
        }
        case device_t::CPU: {
            //        alm_row = mdarray<std::complex<T>, 3>(kp.num_gkvec_row(), max_mt_aw, 1,
            //        H0_.ctx().mem_pool(memory_t::host)); alm_col = mdarray<std::complex<T>, 3>(kp.num_gkvec_col(),
            //        max_mt_aw, 1, H0_.ctx().mem_pool(memory_t::host)); halm_col = mdarray<std::complex<T>,
            //        3>(kp.num_gkvec_col(), max_mt_aw, 1, H0_.ctx().mem_pool(memory_t::host));
            break;
        }
    }

    /* offsets for matching coefficients of individual atoms in the AW block */
    std::vector<int> offsets(uc.num_atoms());

    PROFILE_START("sirius::Hamiltonian_k::set_fv_h_o|zgemm");
    const auto t1 = std::chrono::high_resolution_clock::now();
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

        if (H0_.ctx().cfg().control().print_checksum()) {
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

                // sddk::mdarray<std::complex<T>, 2> alm_row_atom(alm_row.at(memory_t::host, 0, offsets[ia], s),
                //                                               kp.num_gkvec_row(), naw);
                // sddk::mdarray<std::complex<T>, 2> alm_col_atom(alm_col.at(memory_t::host, 0, offsets[ia], s),
                //                                               kp.num_gkvec_col(), naw);
                // sddk::mdarray<std::complex<T>, 2> halm_col_atom(halm_col.at(memory_t::host, 0, offsets[ia], s),
                //                                                kp.num_gkvec_col(), naw);

                sddk::mdarray<std::complex<T>, 2> alm_row_atom;
                sddk::mdarray<std::complex<T>, 2> alm_col_atom;
                sddk::mdarray<std::complex<T>, 2> halm_col_atom;

                switch (pu) {
                    case device_t::CPU: {
                        alm_row_atom = mdarray<std::complex<T>, 2>(alm_row.at(memory_t::host, 0, offsets[ia], s),
                                                                   kp.num_gkvec_row(), naw);

                        alm_col_atom = mdarray<std::complex<T>, 2>(alm_col.at(memory_t::host, 0, offsets[ia], s),
                                                                   kp.num_gkvec_col(), naw);

                        halm_col_atom = mdarray<std::complex<T>, 2>(halm_col.at(memory_t::host, 0, offsets[ia], s),
                                                                    kp.num_gkvec_col(), naw);
                        break;
                    }
                    case device_t::GPU: {
                        alm_row_atom = mdarray<std::complex<T>, 2>(alm_row.at(memory_t::host, 0, offsets[ia], s),
                                                                   alm_row.at(memory_t::device, 0, offsets[ia], s),
                                                                   kp.num_gkvec_row(), naw);

                        alm_col_atom = mdarray<std::complex<T>, 2>(alm_col.at(memory_t::host, 0, offsets[ia], s),
                                                                   alm_col.at(memory_t::device, 0, offsets[ia], s),
                                                                   kp.num_gkvec_col(), naw);

                        halm_col_atom = mdarray<std::complex<T>, 2>(halm_col.at(memory_t::host, 0, offsets[ia], s),
                                                                    halm_col.at(memory_t::device, 0, offsets[ia], s),
                                                                    kp.num_gkvec_col(), naw);
                        break;
                    }
                }

                kp.alm_coeffs_col().template generate<false>(atom, alm_col_atom);
                /* can't copy alm to device how as it might be modified by the iora */

                H0_.template apply_hmt_to_apw<spin_block_t::nm>(atom, kp.num_gkvec_col(), alm_col_atom, halm_col_atom);
                if (pu == device_t::GPU) {
                    halm_col_atom.copy_to(memory_t::device, stream_id(tid));
                }

                /* generate conjugated matching coefficients */
                kp.alm_coeffs_row().template generate<true>(atom, alm_row_atom);
                if (pu == device_t::GPU) {
                    alm_row_atom.copy_to(memory_t::device, stream_id(tid));
                }

                /* setup apw-lo and lo-apw blocks */
                set_fv_h_o_apw_lo(atom, ia, alm_row_atom, alm_col_atom, h__, o__);

                /* finally, modify alm coefficients for iora */
                if (H0_.ctx().valence_relativity() == relativity_t::iora) {
                    // TODO: check if we can modify alm_col with IORA eralier and then not apply it in
                    // set_fv_h_o_apw_lo()
                    H0_.add_o1mt_to_apw(atom, kp.num_gkvec_col(), alm_col_atom);
                }

                if (pu == device_t::GPU) {
                    alm_col_atom.copy_to(memory_t::device, stream_id(tid));
                }
            }
            acc::sync_stream(stream_id(tid));
        }
        // acc::sync_stream(stream_id(omp_get_max_threads()));

        if (H0_.ctx().cfg().control().print_checksum()) {
            std::complex<T> z1 = alm_row.checksum();
            std::complex<T> z2 = alm_col.checksum();
            std::complex<T> z3 = halm_col.checksum();
            utils::print_checksum("alm_row", z1);
            utils::print_checksum("alm_col", z2);
            utils::print_checksum("halm_col", z3);
        }

        linalg(la).gemm('N', 'T', kp.num_gkvec_row(), kp.num_gkvec_col(), num_mt_aw,
                        &linalg_const<std::complex<T>>::one(), alm_row.at(mt1, 0, 0, s), alm_row.ld(),
                        alm_col.at(mt1, 0, 0, s), alm_col.ld(), &linalg_const<std::complex<T>>::one(), o__.at(mt),
                        o__.ld());

        linalg(la).gemm('N', 'T', kp.num_gkvec_row(), kp.num_gkvec_col(), num_mt_aw,
                        &linalg_const<std::complex<T>>::one(), alm_row.at(mt1, 0, 0, s), alm_row.ld(),
                        halm_col.at(mt1, 0, 0, s), halm_col.ld(), &linalg_const<std::complex<T>>::one(), h__.at(mt),
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
    std::chrono::duration<double> tval = std::chrono::high_resolution_clock::now() - t1;
    auto pp                            = utils::get_env<int>("SIRIUS_PRINT_PERFORMANCE");

    if (kp.comm().rank() == 0 && (H0_.ctx().cfg().control().print_performance() || (pp && *pp))) {
        kp.message((pp && *pp) ? 0 : 1, __function_name__, "effective zgemm performance: %12.6f GFlops\n",
                   2 * 8e-9 * kp.num_gkvec() * kp.num_gkvec() * uc.mt_aw_basis_size() / tval.count());
    }

    /* add interstitial contributon */
    set_fv_h_o_it(h__, o__);

    /* setup lo-lo block */
    set_fv_h_o_lo_lo(h__, o__);

    ///*  copy back to GPU */ // TODO: optimize the copys
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
    for (int i = 0; i < kp().num_atom_lo_cols(ia__); i++) {
        int icol = kp().lo_col(ia__, i);
        /* local orbital indices */
        int l     = kp().lo_basis_descriptor_col(icol).l;
        int lm    = kp().lo_basis_descriptor_col(icol).lm;
        int idxrf = kp().lo_basis_descriptor_col(icol).idxrf;
        int order = kp().lo_basis_descriptor_col(icol).order;
        /* loop over apw components and update H */
        for (int j1 = 0; j1 < type.mt_aw_basis_size(); j1++) {
            int lm1    = type.indexb(j1).lm;
            int idxrf1 = type.indexb(j1).idxrf;

            auto zsum = atom__.radial_integrals_sum_L3<spin_block_t::nm>(
                idxrf, idxrf1, H0_.potential().gaunt_coefs().gaunt_vector(lm1, lm));

            if (std::abs(zsum) > 1e-14) {
                for (int igkloc = 0; igkloc < kp().num_gkvec_row(); igkloc++) {
                    h__(igkloc, kp().num_gkvec_col() + icol) +=
                        static_cast<std::complex<T>>(zsum) * alm_row__(igkloc, j1);
                }
            }
        }
        /* update O */
        for (int order1 = 0; order1 < type.aw_order(l); order1++) {
            int xi1 = type.indexb().index_by_lm_order(lm, order1);
            T ori   = atom__.symmetry_class().o_radial_integral(l, order1, order);
            if (H0_.ctx().valence_relativity() == relativity_t::iora) {
                int idxrf1 = type.indexr().index_by_l_order(l, order1);
                ori += atom__.symmetry_class().o1_radial_integral(idxrf1, idxrf);
            }

            for (int igkloc = 0; igkloc < kp().num_gkvec_row(); igkloc++) {
                o__(igkloc, kp().num_gkvec_col() + icol) += ori * alm_row__(igkloc, xi1);
            }
        }
    }

    std::vector<std::complex<T>> ztmp(kp().num_gkvec_col());
    /* lo-apw block */
    for (int i = 0; i < kp().num_atom_lo_rows(ia__); i++) {
        int irow = kp().lo_row(ia__, i);
        /* local orbital indices */
        int l     = kp().lo_basis_descriptor_row(irow).l;
        int lm    = kp().lo_basis_descriptor_row(irow).lm;
        int idxrf = kp().lo_basis_descriptor_row(irow).idxrf;
        int order = kp().lo_basis_descriptor_row(irow).order;

        std::fill(ztmp.begin(), ztmp.end(), 0);

        /* loop over apw components */
        for (int j1 = 0; j1 < type.mt_aw_basis_size(); j1++) {
            int lm1    = type.indexb(j1).lm;
            int idxrf1 = type.indexb(j1).idxrf;

            auto zsum = atom__.radial_integrals_sum_L3<spin_block_t::nm>(
                idxrf1, idxrf, H0_.potential().gaunt_coefs().gaunt_vector(lm, lm1));

            if (std::abs(zsum) > 1e-14) {
                for (int igkloc = 0; igkloc < kp().num_gkvec_col(); igkloc++) {
                    ztmp[igkloc] += static_cast<std::complex<T>>(zsum) * alm_col__(igkloc, j1);
                }
            }
        }

        for (int igkloc = 0; igkloc < kp().num_gkvec_col(); igkloc++) {
            h__(irow + kp().num_gkvec_row(), igkloc) += ztmp[igkloc];
        }

        for (int order1 = 0; order1 < type.aw_order(l); order1++) {
            int xi1 = type.indexb().index_by_lm_order(lm, order1);
            T ori   = atom__.symmetry_class().o_radial_integral(l, order, order1);
            if (H0_.ctx().valence_relativity() == relativity_t::iora) {
                int idxrf1 = type.indexr().index_by_l_order(l, order1);
                ori += atom__.symmetry_class().o1_radial_integral(idxrf, idxrf1);
            }

            for (int igkloc = 0; igkloc < kp().num_gkvec_col(); igkloc++) {
                o__(irow + kp().num_gkvec_row(), igkloc) += ori * alm_col__(igkloc, xi1);
            }
        }
    }
}

template <typename T>
void
Hamiltonian_k<T>::set_fv_h_o_lo_lo(dmatrix<std::complex<T>>& h__, dmatrix<std::complex<T>>& o__) const
{
    PROFILE("sirius::Hamiltonian_k::set_fv_h_o_lo_lo");

    auto& kp = this->kp();

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

                h__(kp.num_gkvec_row() + irow, kp.num_gkvec_col() + icol) +=
                    atom.template radial_integrals_sum_L3<spin_block_t::nm>(
                        idxrf1, idxrf2, H0_.potential().gaunt_coefs().gaunt_vector(lm1, lm2));

                if (lm1 == lm2) {
                    int l      = kp.lo_basis_descriptor_row(irow).l;
                    int order1 = kp.lo_basis_descriptor_row(irow).order;
                    int order2 = kp.lo_basis_descriptor_col(icol).order;
                    o__(kp.num_gkvec_row() + irow, kp.num_gkvec_col() + icol) +=
                        atom.symmetry_class().o_radial_integral(l, order1, order2);
                    if (H0_.ctx().valence_relativity() == relativity_t::iora) {
                        int idxrf1 = atom.type().indexr().index_by_l_order(l, order1);
                        int idxrf2 = atom.type().indexr().index_by_l_order(l, order2);
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
Hamiltonian_k<T>::set_fv_h_o_it(dmatrix<std::complex<T>>& h__, dmatrix<std::complex<T>>& o__) const
{
    PROFILE("sirius::Hamiltonian_k::set_fv_h_o_it");

    double sq_alpha_half = 0.5 * std::pow(speed_of_light, -2);

    auto& kp = this->kp();

    #pragma omp parallel for default(shared)
    for (int igk_col = 0; igk_col < kp.num_gkvec_col(); igk_col++) {
        int ig_col = kp.igk_col(igk_col);
        /* fractional coordinates of G vectors */
        auto gvec_col = kp.gkvec().gvec(ig_col);
        /* Cartesian coordinates of G+k vectors */
        auto gkvec_col_cart = kp.gkvec().template gkvec_cart<index_domain_t::global>(ig_col);
        for (int igk_row = 0; igk_row < kp.num_gkvec_row(); igk_row++) {
            int ig_row          = kp.igk_row(igk_row);
            auto gvec_row       = kp.gkvec().gvec(ig_row);
            auto gkvec_row_cart = kp.gkvec().template gkvec_cart<index_domain_t::global>(ig_row);
            int ig12            = H0().ctx().gvec().index_g12(gvec_row, gvec_col);
            /* pw kinetic energy */
            double t1 = 0.5 * geometry3d::dot(gkvec_row_cart, gkvec_col_cart);

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
                case relativity_t::none: {
                    h__(igk_row, igk_col) += t1 * H0().ctx().theta_pw(ig12);
                    break;
                }
                default: {
                    break;
                }
            }
        }
    }
}

//== template <spin_block_t sblock>
//== void Band::apply_uj_correction(mdarray<double_complex, 2>& fv_states, mdarray<double_complex, 3>& hpsi)
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
//==                                 double_complex z1 = fv_states(offset + idx1, ist) * ori;
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
template <typename F, typename>
void
Hamiltonian_k<T>::apply_h_s(spin_range spins__, int N__, int n__, Wave_functions<T>& phi__, Wave_functions<T>* hphi__,
                            Wave_functions<T>* sphi__)
{
    PROFILE("sirius::Hamiltonian_k::apply_h_s");

    double t1 = -omp_get_wtime();

    if (hphi__ != nullptr) {
        /* apply local part of Hamiltonian */
        H0().local_op().apply_h(reinterpret_cast<spfft_transform_type<T>&>(kp().spfft_transform()),
                                kp().gkvec_partition(), spins__, phi__, *hphi__, N__, n__);
    }

    t1 += omp_get_wtime();

    if (H0().ctx().cfg().control().print_performance()) {
        kp().message(1, __function_name__, "hloc performance: %12.6f bands/sec", n__ / t1);
    }

    if (H0().ctx().cfg().control().print_checksum() && hphi__) {
        for (int ispn : spins__) {
            auto cs1 = phi__.checksum(get_device_t(phi__.preferred_memory_t()), ispn, N__, n__);
            auto cs2 = hphi__->checksum(get_device_t(hphi__->preferred_memory_t()), ispn, N__, n__);
            if (kp().comm().rank() == 0) {
                std::stringstream s;
                s << "phi_" << ispn;
                utils::print_checksum(s.str(), cs1);
                s.str("");
                s << "hphi_" << ispn;
                utils::print_checksum(s.str(), cs2);
            }
        }
    }

    /* set initial sphi */
    if (sphi__ != nullptr) {
        for (int ispn : spins__) {
            sphi__->copy_from(phi__, n__, ispn, N__, ispn, N__);
        }
    }

    /* return if there are no beta-projectors */
    if (H0().ctx().unit_cell().mt_lo_basis_size()) {
        apply_non_local_d_q<F>(spins__, N__, n__, kp().beta_projectors(), phi__, &H0().D(), hphi__, &H0().Q(), sphi__);
    }

    /* apply the hubbard potential if relevant */
    if (H0().ctx().hubbard_correction() && !H0().ctx().gamma_point() && hphi__) {
        /* apply the hubbard potential and deallocate the hubbard wave functions on GPU (if needed) */
        apply_U_operator(H0().ctx(), spins__, N__, n__, kp().hubbard_wave_functions_S(), phi__, this->U(), *hphi__);
    }

    // if ((ctx_.control().print_checksum_) && (hphi__ != nullptr) && (sphi__ != nullptr)) {
    //    for (int ispn = 0; ispn < nsc; ispn++) {
    //        auto cs1 = hphi__->checksum(get_device_t(hphi__->preferred_memory_t()), ispn, N__, n__);
    //        auto cs2 = sphi__->checksum(get_device_t(sphi__->preferred_memory_t()), ispn, N__, n__);
    //        if (kp__->comm().rank() == 0) {
    //            std::stringstream s;
    //            s << "hphi_" << ispn;
    //            utils::print_checksum(s.str(), cs1);
    //            s.str("");
    //            s << "sphi_" << ispn;
    //            utils::print_checksum(s.str(), cs2);
    //        }
    //    }
    //}
}

template <typename T>
void
Hamiltonian_k<T>::apply_fv_h_o(bool apw_only__, bool phi_is_lo__, int N__, int n__, Wave_functions<T>& phi__,
                               Wave_functions<T>* hphi__, Wave_functions<T>* ophi__)
{
    PROFILE("sirius::Hamiltonian_k::apply_fv_h_o");

    /* trivial case */
    if (hphi__ == nullptr && ophi__ == nullptr) {
        return;
    }

    auto& ctx = H0_.ctx();

    /* quick hack: apply O using CPU; GPU is buggy */
    auto pu = (hphi__) ? ctx.processing_unit() : device_t::CPU;

    auto la  = (pu == device_t::CPU) ? linalg_t::blas : linalg_t::gpublas;
    auto mem = (pu == device_t::CPU) ? memory_t::host : memory_t::device;

    // if (ctx.control().print_checksum_) {
    //     phi__.print_checksum(pu, "phi", N__, n__);
    // }

    if (!apw_only__) {
        if (hphi__ != nullptr) {
            /* zero the local-orbital part */
            hphi__->mt_coeffs(0).zero(memory_t::host, N__, n__);
        }
        if (ophi__ != nullptr) {
            /* zero the local-orbital part */
            ophi__->mt_coeffs(0).zero(memory_t::host, N__, n__);
        }
    }

    if (!phi_is_lo__) {
        /* interstitial part */
        H0_.local_op().apply_h_o(reinterpret_cast<spfft_transform_type<T>&>(kp().spfft_transform()),
                                 kp().gkvec_partition(), N__, n__, phi__, hphi__, ophi__);

        // if (ctx.control().print_checksum_) {
        //     if (hphi__) {
        //         hphi__->print_checksum(pu, "hloc_phi", N__, n__);
        //     }
        //     if (ophi__) {
        //         ophi__->print_checksum(pu, "oloc_phi", N__, n__);
        //     }
        // }
    } else {
        /* zero the APW part */
        switch (pu) {
            case device_t::CPU: {
                if (hphi__ != nullptr) {
                    hphi__->pw_coeffs(0).zero(memory_t::host, N__, n__);
                }
                if (ophi__ != nullptr) {
                    ophi__->pw_coeffs(0).zero(memory_t::host, N__, n__);
                }
                break;
            }
            case device_t::GPU: {
                if (hphi__ != nullptr) {
                    hphi__->pw_coeffs(0).zero(memory_t::device, N__, n__);
                }
                if (ophi__ != nullptr) {
                    ophi__->pw_coeffs(0).zero(memory_t::device, N__, n__);
                }
                break;
            }
        }
    }

    if (pu == device_t::GPU && !apw_only__) {
        phi__.mt_coeffs(0).copy_to(memory_t::host, N__, n__);
    }

    /* short name for local number of G+k vectors */
    int ngv = kp().num_gkvec_loc();

    /* split atoms in blocks */
    int num_atoms_in_block = 2 * omp_get_max_threads();

    /* number of blocks of atoms */
    int nblk = utils::num_blocks(ctx.unit_cell().num_atoms(), num_atoms_in_block);

    /* maximum number of AW radial functions in a block of atoms */
    int max_mt_aw = num_atoms_in_block * ctx.unit_cell().max_mt_aw_basis_size();

    /* maximum number of LO radial functions in a block of atoms */
    int max_mt_lo = num_atoms_in_block * ctx.unit_cell().max_mt_lo_basis_size();

    PROFILE_START("sirius::Hamiltonian_k::apply_fv_h_o|alloc");

    /* matching coefficients for a block of atoms */
    matrix<std::complex<T>> alm_block;
    matrix<std::complex<T>> halm_block;

    switch (pu) {
        case device_t::CPU: {
            alm_block  = matrix<std::complex<T>>(ngv, max_mt_aw, ctx.mem_pool(memory_t::host));
            halm_block = matrix<std::complex<T>>(ngv, std::max(max_mt_aw, max_mt_lo), ctx.mem_pool(memory_t::host));
            break;
        }
        case device_t::GPU: {
            alm_block = matrix<std::complex<T>>(ngv, max_mt_aw, ctx.mem_pool(memory_t::host_pinned));
            alm_block.allocate(ctx.mem_pool(memory_t::device));
            halm_block =
                matrix<std::complex<T>>(ngv, std::max(max_mt_aw, max_mt_lo), ctx.mem_pool(memory_t::host_pinned));
            halm_block.allocate(ctx.mem_pool(memory_t::device));
            break;
        }
    }
    size_t sz = max_mt_aw * n__;
    /* buffers for alm_phi and halm_phi */
    mdarray<std::complex<T>, 1> alm_phi_buf;
    if (ophi__ != nullptr) {
        switch (pu) {
            case device_t::CPU: {
                alm_phi_buf = mdarray<std::complex<T>, 1>(sz, ctx.mem_pool(memory_t::host));
                break;
            }
            case device_t::GPU: {
                alm_phi_buf = mdarray<std::complex<T>, 1>(sz, ctx.mem_pool(memory_t::host_pinned));
                alm_phi_buf.allocate(ctx.mem_pool(memory_t::device));
                break;
            }
        }
    }
    mdarray<std::complex<T>, 1> halm_phi_buf;
    if (hphi__ != nullptr) {
        switch (pu) {
            case device_t::CPU: {
                halm_phi_buf = mdarray<std::complex<T>, 1>(sz, ctx.mem_pool(memory_t::host));
                break;
            }
            case device_t::GPU: {
                size_t sz    = max_mt_aw * n__;
                halm_phi_buf = mdarray<std::complex<T>, 1>(sz, ctx.mem_pool(memory_t::host_pinned));
                halm_phi_buf.allocate(ctx.mem_pool(memory_t::device));
                break;
            }
        }
    }
    PROFILE_STOP("sirius::Hamiltonian_k::apply_fv_h_o|alloc");

    /* generate matching coefficients Alm(G+k) for a block of atoms */
    auto generate_alm = [&](int atom_begin, int atom_end, std::vector<int>& offsets_aw) {
        PROFILE("sirius::Hamiltonian_k::apply_fv_h_o|alm");
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            for (int ia = atom_begin; ia < atom_end; ia++) {
                if (ia % omp_get_num_threads() == tid) {
                    int ialoc  = ia - atom_begin;
                    auto& atom = ctx.unit_cell().atom(ia);
                    auto& type = atom.type();

                    /* wrapper for matching coefficients for a given atom */
                    mdarray<std::complex<T>, 2> alm_tmp;
                    mdarray<std::complex<T>, 2> halm_tmp;
                    switch (pu) {
                        case device_t::CPU: {
                            alm_tmp = mdarray<std::complex<T>, 2>(alm_block.at(memory_t::host, 0, offsets_aw[ialoc]),
                                                                  ngv, type.mt_aw_basis_size());
                            if (hphi__ != nullptr) {
                                halm_tmp = mdarray<std::complex<T>, 2>(
                                    halm_block.at(memory_t::host, 0, offsets_aw[ialoc]), ngv, type.mt_aw_basis_size());
                            }
                            break;
                        }
                        case device_t::GPU: {
                            alm_tmp = mdarray<std::complex<T>, 2>(alm_block.at(memory_t::host, 0, offsets_aw[ialoc]),
                                                                  alm_block.at(memory_t::device, 0, offsets_aw[ialoc]),
                                                                  ngv, type.mt_aw_basis_size());
                            if (hphi__ != nullptr) {
                                halm_tmp =
                                    mdarray<std::complex<T>, 2>(halm_block.at(memory_t::host, 0, offsets_aw[ialoc]),
                                                                halm_block.at(memory_t::device, 0, offsets_aw[ialoc]),
                                                                ngv, type.mt_aw_basis_size());
                            }
                            break;
                        }
                    }

                    /* generate LAPW matching coefficients on the CPU */
                    kp().alm_coeffs_loc().template generate<true>(atom, alm_tmp);
                    if (pu == device_t::GPU) {
                        alm_tmp.copy_to(memory_t::device, stream_id(tid));
                    }
                    if (hphi__ != nullptr) {
                        H0_.template apply_hmt_to_apw<spin_block_t::nm>(atom, ngv, alm_tmp, halm_tmp);
                        if (pu == device_t::GPU) {
                            halm_tmp.copy_to(memory_t::device, stream_id(tid));
                        }
                    }
                }
            }
            if (pu == device_t::GPU) {
                acc::sync_stream(stream_id(tid));
            }
        }
    };

    auto compute_alm_phi = [&](matrix<std::complex<T>>& alm_phi, matrix<std::complex<T>>& halm_phi, int num_mt_aw) {
        PROFILE("sirius::Hamiltonian_k::apply_fv_h_o|alm_phi");

        /* first zgemm: A(G, lm)^{T} * C(G, i) and  hA(G, lm)^{T} * C(G, i) */
        switch (pu) {
            case device_t::CPU: {
                if (ophi__ != nullptr) {
                    /* create resulting array with proper dimensions from the already allocated chunk of memory */
                    alm_phi = matrix<std::complex<T>>(alm_phi_buf.at(memory_t::host), num_mt_aw, n__);
                    /* alm_phi(lm, i) = A(G, lm)^{T} * C(G, i), remember that Alm was conjugated */
                    linalg(linalg_t::blas)
                        .gemm('C', 'N', num_mt_aw, n__, ngv, &linalg_const<std::complex<T>>::one(),
                              alm_block.at(memory_t::host), alm_block.ld(),
                              phi__.pw_coeffs(0).prime().at(memory_t::host, 0, N__), phi__.pw_coeffs(0).prime().ld(),
                              &linalg_const<std::complex<T>>::zero(), alm_phi.at(memory_t::host), alm_phi.ld());
                }
                if (hphi__ != nullptr) {
                    /* create resulting array with proper dimensions from the already allocated chunk of memory */
                    halm_phi = matrix<std::complex<T>>(halm_phi_buf.at(memory_t::host), num_mt_aw, n__);
                    /* halm_phi(lm, i) = H_{mt}A(G, lm)^{T} * C(G, i) */
                    linalg(linalg_t::blas)
                        .gemm('C', 'N', num_mt_aw, n__, ngv, &linalg_const<std::complex<T>>::one(),
                              halm_block.at(memory_t::host), halm_block.ld(),
                              phi__.pw_coeffs(0).prime().at(memory_t::host, 0, N__), phi__.pw_coeffs(0).prime().ld(),
                              &linalg_const<std::complex<T>>::zero(), halm_phi.at(memory_t::host), halm_phi.ld());
                }
                break;
            }
            case device_t::GPU: {
                if (ophi__ != nullptr) {
                    /* create resulting array with proper dimensions from the already allocated chunk of memory */
                    alm_phi = matrix<std::complex<T>>(alm_phi_buf.at(memory_t::host), alm_phi_buf.at(memory_t::device),
                                                      num_mt_aw, n__);
                    /* alm_phi(lm, i) = A(G, lm)^{T} * C(G, i) */
                    linalg(linalg_t::gpublas)
                        .gemm('C', 'N', num_mt_aw, n__, ngv, &linalg_const<std::complex<T>>::one(),
                              alm_block.at(memory_t::device), alm_block.ld(),
                              phi__.pw_coeffs(0).prime().at(memory_t::device, 0, N__), phi__.pw_coeffs(0).prime().ld(),
                              &linalg_const<std::complex<T>>::zero(), alm_phi.at(memory_t::device), alm_phi.ld());
                    alm_phi.copy_to(memory_t::host);
                }
                if (hphi__ != nullptr) {
                    /* create resulting array with proper dimensions from the already allocated chunk of memory */
                    halm_phi = matrix<std::complex<T>>(halm_phi_buf.at(memory_t::host),
                                                       halm_phi_buf.at(memory_t::device), num_mt_aw, n__);
                    /* halm_phi(lm, i) = H_{mt}A(G, lm)^{T} * C(G, i) */
                    linalg(linalg_t::gpublas)
                        .gemm('C', 'N', num_mt_aw, n__, ngv, &linalg_const<std::complex<T>>::one(),
                              halm_block.at(memory_t::device), halm_block.ld(),
                              phi__.pw_coeffs(0).prime().at(memory_t::device, 0, N__), phi__.pw_coeffs(0).prime().ld(),
                              &linalg_const<std::complex<T>>::zero(), halm_phi.at(memory_t::device), halm_phi.ld());

                    halm_phi.copy_to(memory_t::host);
                }
                break;
            }
        }

        if (hphi__ != nullptr) {
            kp().comm().allreduce(halm_phi.at(memory_t::host), num_mt_aw * n__);
            if (pu == device_t::GPU) {
                halm_phi.copy_to(memory_t::device);
            }
        }

        if (ophi__ != nullptr) {
            kp().comm().allreduce(alm_phi.at(memory_t::host), num_mt_aw * n__);
            if (pu == device_t::GPU) {
                alm_phi.copy_to(memory_t::device);
            }
        }
    };

    auto compute_apw_apw = [&](matrix<std::complex<T>>& alm_phi, matrix<std::complex<T>>& halm_phi, int num_mt_aw) {
        PROFILE("sirius::Hamiltonian_k::apply_fv_h_o|apw-apw");
        auto la = linalg_t::none;
        auto mt = memory_t::none;
        switch (pu) {
            case device_t::CPU: {
                la = linalg_t::blas;
                mt = memory_t::host;
                break;
            }
            case device_t::GPU: {
                la = linalg_t::gpublas;
                mt = memory_t::device;
                break;
            }
        }
        /* second zgemm: Alm^{*} (Alm * C) */
        if (ophi__ != nullptr) {
            /* APW-APW contribution to overlap */
            linalg(la).gemm('N', 'N', ngv, n__, num_mt_aw, &linalg_const<std::complex<T>>::one(), alm_block.at(mt),
                            alm_block.ld(), alm_phi.at(mt), alm_phi.ld(), &linalg_const<std::complex<T>>::one(),
                            ophi__->pw_coeffs(0).prime().at(mt, 0, N__), ophi__->pw_coeffs(0).prime().ld());
        }
        if (hphi__ != nullptr) {
            /* APW-APW contribution to Hamiltonian */
            linalg(la).gemm('N', 'N', ngv, n__, num_mt_aw, &linalg_const<std::complex<T>>::one(), alm_block.at(mt),
                            alm_block.ld(), halm_phi.at(mt), halm_phi.ld(), &linalg_const<std::complex<T>>::one(),
                            hphi__->pw_coeffs(0).prime().at(mt, 0, N__), hphi__->pw_coeffs(0).prime().ld());
        }
    };

    auto collect_lo = [&](int atom_begin, int atom_end, std::vector<int>& offsets_lo,
                          matrix<std::complex<T>>& phi_lo_block) {
        PROFILE("sirius::Hamiltonian_k::apply_fv_h_o|phi_lo");
        /* broadcast local orbital coefficients */
        for (int ia = atom_begin; ia < atom_end; ia++) {
            int ialoc        = ia - atom_begin;
            auto& atom       = ctx.unit_cell().atom(ia);
            auto& type       = atom.type();
            auto ia_location = phi__.spl_num_atoms().location(ia);

            /* lo coefficients for a given atom and all bands */
            matrix<std::complex<T>> phi_lo_ia(type.mt_lo_basis_size(), n__);

            if (ia_location.rank == kp().comm().rank()) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < n__; i++) {
                    std::memcpy(&phi_lo_ia(0, i),
                                phi__.mt_coeffs(0).prime().at(memory_t::host,
                                                              phi__.offset_mt_coeffs(ia_location.local_index), N__ + i),
                                type.mt_lo_basis_size() * sizeof(std::complex<T>));
                }
            }
            /* broadcast from a rank */
            kp().comm().bcast(phi_lo_ia.at(memory_t::host), static_cast<int>(phi_lo_ia.size()), ia_location.rank);
            /* wrtite into a proper position in a block */
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n__; i++) {
                std::memcpy(&phi_lo_block(offsets_lo[ialoc], i), &phi_lo_ia(0, i),
                            type.mt_lo_basis_size() * sizeof(std::complex<T>));
            }
        } // ia

        if (pu == device_t::GPU) {
            phi_lo_block.copy_to(memory_t::device);
        }
    };

    auto compute_apw_lo = [&](int atom_begin, int atom_end, int num_mt_lo, std::vector<int>& offsets_aw,
                              std::vector<int> offsets_lo, matrix<std::complex<T>>& phi_lo_block) {
        PROFILE("sirius::Hamiltonian_k::apply_fv_h_o|apw-lo");
        /* apw-lo block for hphi */
        if (hphi__ != nullptr) {
            for (int ia = atom_begin; ia < atom_end; ia++) {
                int ialoc  = ia - atom_begin;
                auto& atom = ctx.unit_cell().atom(ia);
                auto& type = atom.type();
                int naw    = type.mt_aw_basis_size();
                int nlo    = type.mt_lo_basis_size();

                matrix<std::complex<T>> hmt(naw, nlo);
                #pragma omp parallel for schedule(static)
                for (int ilo = 0; ilo < nlo; ilo++) {
                    int xi_lo = naw + ilo;
                    /* local orbital indices */
                    int lm_lo    = type.indexb(xi_lo).lm;
                    int idxrf_lo = type.indexb(xi_lo).idxrf;
                    for (int xi = 0; xi < naw; xi++) {
                        int lm_aw    = type.indexb(xi).lm;
                        int idxrf_aw = type.indexb(xi).idxrf;
                        auto& gc     = H0_.potential().gaunt_coefs().gaunt_vector(lm_aw, lm_lo);
                        hmt(xi, ilo) = atom.template radial_integrals_sum_L3<spin_block_t::nm>(idxrf_aw, idxrf_lo, gc);
                    }
                }
                if (pu == device_t::GPU) {
                    hmt.allocate(memory_t::device).copy_to(memory_t::device);
                }
                linalg(la).gemm('N', 'N', ngv, nlo, naw, &linalg_const<std::complex<T>>::one(),
                                alm_block.at(mem, 0, offsets_aw[ialoc]), alm_block.ld(), hmt.at(mem), hmt.ld(),
                                &linalg_const<std::complex<T>>::zero(), halm_block.at(mem, 0, offsets_lo[ialoc]),
                                halm_block.ld());
            } // ia
            linalg(la).gemm('N', 'N', ngv, n__, num_mt_lo, &linalg_const<std::complex<T>>::one(), halm_block.at(mem),
                            halm_block.ld(), phi_lo_block.at(mem), phi_lo_block.ld(),
                            &linalg_const<std::complex<T>>::one(), hphi__->pw_coeffs(0).prime().at(mem, 0, N__),
                            hphi__->pw_coeffs(0).prime().ld());
        }

        /* apw-lo block for ophi */
        if (ophi__ != nullptr) {
            halm_block.zero();
            for (int ia = atom_begin; ia < atom_end; ia++) {
                int ialoc  = ia - atom_begin;
                auto& atom = ctx.unit_cell().atom(ia);
                auto& type = atom.type();
                int naw    = type.mt_aw_basis_size();
                int nlo    = type.mt_lo_basis_size();

                #pragma omp parallel for schedule(static)
                for (int ilo = 0; ilo < nlo; ilo++) {
                    int xi_lo = naw + ilo;
                    /* local orbital indices */
                    int l_lo     = type.indexb(xi_lo).l;
                    int lm_lo    = type.indexb(xi_lo).lm;
                    int order_lo = type.indexb(xi_lo).order;
                    /* use halm as temporary buffer to compute alm*o */
                    for (int order_aw = 0; order_aw < (int)type.aw_descriptor(l_lo).size(); order_aw++) {
                        for (int igloc = 0; igloc < ngv; igloc++) {
                            halm_block(igloc, offsets_lo[ialoc] + ilo) +=
                                alm_block(igloc, offsets_aw[ialoc] + type.indexb_by_lm_order(lm_lo, order_aw)) *
                                static_cast<T>(atom.symmetry_class().o_radial_integral(l_lo, order_aw, order_lo));
                        } // TODO: block copy to GPU
                    }
                }
            } // ia
            if (pu == device_t::GPU) {
                halm_block.copy_to(memory_t::device, 0, ngv * num_mt_lo);
            }
            linalg(la).gemm('N', 'N', ngv, n__, num_mt_lo, &linalg_const<std::complex<T>>::one(), halm_block.at(mem),
                            halm_block.ld(), phi_lo_block.at(mem), phi_lo_block.ld(),
                            &linalg_const<std::complex<T>>::one(), ophi__->pw_coeffs(0).prime().at(mem, 0, N__),
                            ophi__->pw_coeffs(0).prime().ld());
        }
    };

    PROFILE_START("sirius::Hamiltonian_k::apply_fv_h_o|mt");
    /* loop over blocks of atoms */
    for (int iblk = 0; iblk < nblk; iblk++) {
        int atom_begin = iblk * num_atoms_in_block;
        int atom_end   = std::min(ctx.unit_cell().num_atoms(), (iblk + 1) * num_atoms_in_block);
        /* actual number of AW radial functions in a block of atoms */
        int num_mt_aw{0};
        /* actual number of local orbitals in a block of atoms */
        int num_mt_lo{0};
        std::vector<int> offsets_aw(num_atoms_in_block);
        std::vector<int> offsets_lo(num_atoms_in_block);
        for (int ia = atom_begin; ia < atom_end; ia++) {
            int ialoc         = ia - atom_begin;
            auto& atom        = ctx.unit_cell().atom(ia);
            auto& type        = atom.type();
            offsets_aw[ialoc] = num_mt_aw;
            offsets_lo[ialoc] = num_mt_lo;
            num_mt_aw += type.mt_aw_basis_size();
            num_mt_lo += type.mt_lo_basis_size();
        }

        /* created alm and halm for a block of atoms */
        generate_alm(atom_begin, atom_end, offsets_aw);

        matrix<std::complex<T>> alm_phi;
        matrix<std::complex<T>> halm_phi;

        compute_alm_phi(alm_phi, halm_phi, num_mt_aw);

        if (!phi_is_lo__) {
            compute_apw_apw(alm_phi, halm_phi, num_mt_aw);
        }

        if (!apw_only__ && num_mt_lo) {
            /* local orbital coefficients for a block of atoms and all states */
            matrix<std::complex<T>> phi_lo_block(num_mt_lo, n__);
            if (pu == device_t::GPU) {
                phi_lo_block.allocate(memory_t::device);
            }
            collect_lo(atom_begin, atom_end, offsets_lo, phi_lo_block);

            compute_apw_lo(atom_begin, atom_end, num_mt_lo, offsets_aw, offsets_lo, phi_lo_block);

            PROFILE_START("sirius::Hamiltonian::apply_fv_h_o|lo-apw");
            /* lo-APW contribution */
            for (int ia = atom_begin; ia < atom_end; ia++) {
                int ialoc  = ia - atom_begin;
                auto& atom = ctx.unit_cell().atom(ia);
                auto& type = atom.type();

                auto ia_location = phi__.spl_num_atoms().location(ia);

                if (ia_location.rank == kp().comm().rank()) {
                    int offset_mt_coeffs = phi__.offset_mt_coeffs(ia_location.local_index);

                    #pragma omp parallel for schedule(static)
                    for (int ilo = 0; ilo < type.mt_lo_basis_size(); ilo++) {
                        int xi_lo = type.mt_aw_basis_size() + ilo;
                        /* local orbital indices */
                        int l_lo     = type.indexb(xi_lo).l;
                        int lm_lo    = type.indexb(xi_lo).lm;
                        int order_lo = type.indexb(xi_lo).order;
                        int idxrf_lo = type.indexb(xi_lo).idxrf;

                        /* lo-lo contribution */
                        for (int jlo = 0; jlo < type.mt_lo_basis_size(); jlo++) {
                            int xi_lo1 = type.mt_aw_basis_size() + jlo;
                            int lm1    = type.indexb(xi_lo1).lm;
                            int order1 = type.indexb(xi_lo1).order;
                            int idxrf1 = type.indexb(xi_lo1).idxrf;
                            if (lm_lo == lm1 && ophi__ != nullptr) {
                                for (int i = 0; i < n__; i++) {
                                    ophi__->mt_coeffs(0).prime(offset_mt_coeffs + ilo, N__ + i) +=
                                        phi_lo_block(offsets_lo[ialoc] + jlo, i) *
                                        static_cast<T>(atom.symmetry_class().o_radial_integral(l_lo, order_lo, order1));
                                }
                            }
                            if (hphi__ != nullptr) {
                                auto& gc = H0_.potential().gaunt_coefs().gaunt_vector(lm_lo, lm1);
                                for (int i = 0; i < n__; i++) {
                                    hphi__->mt_coeffs(0).prime(offset_mt_coeffs + ilo, N__ + i) +=
                                        phi_lo_block(offsets_lo[ialoc] + jlo, i) *
                                        static_cast<std::complex<T>>(
                                            atom.template radial_integrals_sum_L3<spin_block_t::nm>(idxrf_lo, idxrf1,
                                                                                                    gc));
                                }
                            }
                        }

                        /* lo-APW contribution */
                        if (!phi_is_lo__) {
                            if (ophi__ != nullptr) {
                                for (int i = 0; i < n__; i++) {
                                    /* lo-APW contribution to ophi */
                                    for (int order_aw = 0; order_aw < (int)type.aw_descriptor(l_lo).size();
                                         order_aw++) {
                                        ophi__->mt_coeffs(0).prime(offset_mt_coeffs + ilo, N__ + i) +=
                                            static_cast<T>(
                                                atom.symmetry_class().o_radial_integral(l_lo, order_lo, order_aw)) *
                                            alm_phi(offsets_aw[ialoc] + type.indexb_by_lm_order(lm_lo, order_aw), i);
                                    }
                                }
                            }

                            if (hphi__ != nullptr) {
                                for (int i = 0; i < n__; i++) {
                                    std::complex<T> z(0, 0);
                                    for (int xi = 0; xi < type.mt_aw_basis_size(); xi++) {
                                        int lm_aw    = type.indexb(xi).lm;
                                        int idxrf_aw = type.indexb(xi).idxrf;
                                        auto& gc     = H0_.potential().gaunt_coefs().gaunt_vector(lm_lo, lm_aw);
                                        z += static_cast<std::complex<T>>(
                                                 atom.template radial_integrals_sum_L3<spin_block_t::nm>(
                                                     idxrf_lo, idxrf_aw, gc)) *
                                             alm_phi(offsets_aw[ialoc] + xi, i);
                                    }
                                    /* lo-APW contribution to hphi */
                                    hphi__->mt_coeffs(0).prime(offset_mt_coeffs + ilo, N__ + i) += z;
                                }
                            }
                        }
                    }
                }
            }
            PROFILE_STOP("sirius::Hamiltonian::apply_fv_h_o|lo-apw");
        }
    }

    PROFILE_STOP("sirius::Hamiltonian_k::apply_fv_h_o|mt");
    if (pu == device_t::GPU && !apw_only__) {
        if (hphi__ != nullptr) {
            hphi__->mt_coeffs(0).copy_to(memory_t::device, N__, n__);
        }
        if (ophi__ != nullptr) {
            ophi__->mt_coeffs(0).copy_to(memory_t::device, N__, n__);
        }
    }
    // if (ctx.control().print_checksum_) {
    //     if (hphi__) {
    //         hphi__->print_checksum(pu, "hphi", N__, n__);
    //     }
    //     if (ophi__) {
    //         ophi__->print_checksum(pu, "ophi", N__, n__);
    //     }
    // }
}

template <typename T>
void
Hamiltonian_k<T>::apply_b(Wave_functions<T>& psi__, std::vector<Wave_functions<T>>& bpsi__)
{
    PROFILE("sirius::Hamiltonian_k::apply_b");

    assert(bpsi__.size() == 2 || bpsi__.size() == 3);

    H0().local_op().apply_b(reinterpret_cast<spfft_transform_type<T>&>(kp().spfft_transform()), 0,
                            H0().ctx().num_fv_states(), psi__, bpsi__);
    H0().apply_bmt(psi__, bpsi__);

    /* copy Bz|\psi> to -Bz|\psi> */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < H0().ctx().num_fv_states(); i++) {
        for (int j = 0; j < psi__.pw_coeffs(0).num_rows_loc(); j++) {
            bpsi__[1].pw_coeffs(0).prime(j, i) = -bpsi__[0].pw_coeffs(0).prime(j, i);
        }
        for (int j = 0; j < psi__.mt_coeffs(0).num_rows_loc(); j++) {
            bpsi__[1].mt_coeffs(0).prime(j, i) = -bpsi__[0].mt_coeffs(0).prime(j, i);
        }
    }
}

template class Hamiltonian_k<double>;

template void Hamiltonian_k<double>::apply_h_s<double>(spin_range spins__, int N__, int n__,
                                                       Wave_functions<double>& phi__, Wave_functions<double>* hphi__,
                                                       Wave_functions<double>* sphi__);

template void Hamiltonian_k<double>::apply_h_s<double_complex>(spin_range spins__, int N__, int n__,
                                                               Wave_functions<double>& phi__,
                                                               Wave_functions<double>* hphi__,
                                                               Wave_functions<double>* sphi__);

template std::pair<mdarray<double, 2>, mdarray<double, 2>> Hamiltonian_k<double>::get_h_o_diag_pw<double, 1>() const;

template std::pair<mdarray<double, 2>, mdarray<double, 2>> Hamiltonian_k<double>::get_h_o_diag_pw<double, 2>() const;

template std::pair<mdarray<double, 2>, mdarray<double, 2>> Hamiltonian_k<double>::get_h_o_diag_pw<double, 3>() const;

template std::pair<mdarray<double, 2>, mdarray<double, 2>>
Hamiltonian_k<double>::get_h_o_diag_pw<double_complex, 1>() const;

template std::pair<mdarray<double, 2>, mdarray<double, 2>>
Hamiltonian_k<double>::get_h_o_diag_pw<double_complex, 2>() const;

template std::pair<mdarray<double, 2>, mdarray<double, 2>>
Hamiltonian_k<double>::get_h_o_diag_pw<double_complex, 3>() const;

template std::pair<mdarray<double, 2>, mdarray<double, 2>> Hamiltonian_k<double>::get_h_o_diag_lapw<1>() const;

template std::pair<mdarray<double, 2>, mdarray<double, 2>> Hamiltonian_k<double>::get_h_o_diag_lapw<2>() const;

template std::pair<mdarray<double, 2>, mdarray<double, 2>> Hamiltonian_k<double>::get_h_o_diag_lapw<3>() const;

#ifdef USE_FP32
template class Hamiltonian_k<float>;

template void Hamiltonian_k<float>::apply_h_s<float>(spin_range spins__, int N__, int n__, Wave_functions<float>& phi__,
                                                     Wave_functions<float>* hphi__, Wave_functions<float>* sphi__);

template void Hamiltonian_k<float>::apply_h_s<std::complex<float>>(spin_range spins__, int N__, int n__,
                                                                   Wave_functions<float>& phi__,
                                                                   Wave_functions<float>* hphi__,
                                                                   Wave_functions<float>* sphi__);

template std::pair<mdarray<float, 2>, mdarray<float, 2>> Hamiltonian_k<float>::get_h_o_diag_pw<float, 1>() const;

template std::pair<mdarray<float, 2>, mdarray<float, 2>> Hamiltonian_k<float>::get_h_o_diag_pw<float, 2>() const;

template std::pair<mdarray<float, 2>, mdarray<float, 2>> Hamiltonian_k<float>::get_h_o_diag_pw<float, 3>() const;

template std::pair<mdarray<float, 2>, mdarray<float, 2>>
Hamiltonian_k<float>::get_h_o_diag_pw<std::complex<float>, 1>() const;

template std::pair<mdarray<float, 2>, mdarray<float, 2>>
Hamiltonian_k<float>::get_h_o_diag_pw<std::complex<float>, 2>() const;

template std::pair<mdarray<float, 2>, mdarray<float, 2>>
Hamiltonian_k<float>::get_h_o_diag_pw<std::complex<float>, 3>() const;

template std::pair<mdarray<float, 2>, mdarray<float, 2>> Hamiltonian_k<float>::get_h_o_diag_lapw<1>() const;

template std::pair<mdarray<float, 2>, mdarray<float, 2>> Hamiltonian_k<float>::get_h_o_diag_lapw<2>() const;

template std::pair<mdarray<float, 2>, mdarray<float, 2>> Hamiltonian_k<float>::get_h_o_diag_lapw<3>() const;
#endif
} // namespace sirius
