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

#include "Hamiltonian/hamiltonian.hpp"

namespace sirius {

/** \file apply.hpp
 *
 *  \brief Contains implementation of various sirius::Hamiltonian apply() functions.
 */


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

/** \param [in]  ispn Index of spin.
 *  \param [in]  N    Starting index of wave-functions.
 *  \param [in]  n    Number of wave-functions to which H and S are applied.
 *  \param [in]  phi  Input wave-functions [storage: CPU && GPU].
 *  \param [out] hphi Hamiltonian, applied to wave-functions [storage: CPU || GPU].
 *  \param [out] sphi Overlap operator, applied to wave-functions [storage: CPU || GPU].
 *
 *  In non-collinear case (ispn = 2) the Hamiltonian and S operator are applied to both components of spinor
 *  wave-functions. Otherwise they are applied to a single component.
 */
template <typename T>
void Hamiltonian_k::apply_h_s(spin_range spins__, int N__, int n__, Wave_functions& phi__, Wave_functions* hphi__,
                              Wave_functions* sphi__)
{
    PROFILE("sirius::Hamiltonian_k::apply_h_s");

    double t1 = -omp_get_wtime();

    if (hphi__ != nullptr) {
        /* apply local part of Hamiltonian */
        H0().local_op().apply_h(kp().spfft_transform(), spins__, phi__, *hphi__, N__, n__);
    }

    t1 += omp_get_wtime();

    if (H0().ctx().control().print_performance_) {
        kp().message(1, __func__, "hloc performace: %12.6f bands/sec", n__ / t1);
    }

    if (H0().ctx().control().print_checksum_ && hphi__) {
        for (int ispn: spins__) {
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

    /* set intial sphi */
    if (sphi__ != nullptr) {
        for (int ispn: spins__) {
            sphi__->copy_from(phi__, n__, ispn, N__, ispn, N__);
        }
    }

    /* return if there are no beta-projectors */
    if (H0().ctx().unit_cell().mt_lo_basis_size()) {
        apply_non_local_d_q<T>(spins__, N__, n__, kp().beta_projectors(), phi__, &H0().D(), hphi__, &H0().Q(), sphi__);
    }

    /* apply the hubbard potential if relevant */
     if (H0().ctx().hubbard_correction() && !H0().ctx().gamma_point() && hphi__) {

       // copy the hubbard wave functions on GPU (if needed) and
       // return afterwards, or if they are not already calculated
       // compute the wave functions and copy them on GPU (if needed)

        //this->U().generate_atomic_orbitals(*kp__, Q());

        // Apply the hubbard potential and deallocate the hubbard wave
        // functions on GPU (if needed)
        H0().potential().U().apply_hubbard_potential(kp().hubbard_wave_functions(), spins__(), N__, n__, phi__, *hphi__);

        //if (ctx_.processing_unit() == device_t::GPU) {
        //    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        //        kp__->hubbard_wave_functions().deallocate(spin_idx(ispn), memory_t::device);
        //    }
        //}
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

void Hamiltonian_k::apply_fv_h_o(bool apw_only__, bool phi_is_lo__, int N__, int n__,
                                 Wave_functions& phi__, Wave_functions* hphi__, Wave_functions* ophi__)
{
    PROFILE("sirius::Hamiltonian_k::apply_fv_h_o");

    /* trivial case */
    if (hphi__ == nullptr && ophi__ == nullptr) {
        return;
    }

    auto& ctx = H0_.ctx();

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
        H0_.local_op().apply_h_o(kp().spfft_transform(), N__, n__, phi__, hphi__, ophi__);
    } else {
        /* zero the APW part */
        switch (ctx.processing_unit()) {
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

    if (ctx.processing_unit() == device_t::GPU && !apw_only__) {
        phi__.mt_coeffs(0).copy_to(memory_t::host, N__, n__);
    }

    /* short name for local number of G+k vectors */
    int ngv = kp().num_gkvec_loc();

    /* split atoms in blocks */
    int num_atoms_in_block = 2 * omp_get_max_threads();
    int nblk               = utils::num_blocks(ctx.unit_cell().num_atoms(), num_atoms_in_block);

    /* maximum number of AW radial functions in a block of atoms */
    int max_mt_aw = num_atoms_in_block * ctx.unit_cell().max_mt_aw_basis_size();
    /* maximum number of LO radial functions in a block of atoms */
    int max_mt_lo = num_atoms_in_block * ctx.unit_cell().max_mt_lo_basis_size();

    utils::timer t0("sirius::Hamiltonian_k::apply_fv_h_o|alloc");

    /* matching coefficients for a block of atoms */
    matrix<double_complex> alm_block;
    matrix<double_complex> halm_block;

    switch (ctx.processing_unit()) {
        case device_t::CPU: {
            alm_block = matrix<double_complex>(ctx.mem_pool(memory_t::host), ngv, max_mt_aw);
            if (hphi__ != nullptr) {
                halm_block = matrix<double_complex>(ctx.mem_pool(memory_t::host), ngv, std::max(max_mt_aw, max_mt_lo));
            }
            break;
        }
        case device_t::GPU: {
            alm_block = matrix<double_complex>(ctx.mem_pool(memory_t::host_pinned), ngv, max_mt_aw);
            alm_block.allocate(ctx.mem_pool(memory_t::device));
            if (hphi__ != nullptr) {
                halm_block =
                    matrix<double_complex>(ctx.mem_pool(memory_t::host_pinned), ngv, std::max(max_mt_aw, max_mt_lo));
                halm_block.allocate(ctx.mem_pool(memory_t::device));
            }
            break;
        }
    }
    size_t sz = max_mt_aw * n__;
    /* buffers for alm_phi and halm_phi */
    mdarray<double_complex, 1> alm_phi_buf;
    if (ophi__ != nullptr) {
        switch (ctx.processing_unit()) {
            case device_t::CPU: {
                alm_phi_buf = mdarray<double_complex, 1>(ctx.mem_pool(memory_t::host), sz);
                break;
            }
            case device_t::GPU: {
                alm_phi_buf = mdarray<double_complex, 1>(ctx.mem_pool(memory_t::host_pinned), sz);
                alm_phi_buf.allocate(ctx.mem_pool(memory_t::device));
                break;
            }
        }
    }
    mdarray<double_complex, 1> halm_phi_buf;
    if (hphi__ != nullptr) {
        switch (ctx.processing_unit()) {
            case device_t::CPU: {
                halm_phi_buf = mdarray<double_complex, 1>(ctx.mem_pool(memory_t::host), sz);
                break;
            }
            case device_t::GPU: {
                size_t sz    = max_mt_aw * n__;
                halm_phi_buf = mdarray<double_complex, 1>(ctx.mem_pool(memory_t::host_pinned), sz);
                halm_phi_buf.allocate(ctx.mem_pool(memory_t::device));
                break;
            }
        }
    }
    t0.stop();

    auto generate_alm = [&](int atom_begin, int atom_end, std::vector<int>& offsets_aw) {
        utils::timer t1("sirius::Hamiltonian_k::apply_fv_h_o|alm");
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            for (int ia = atom_begin; ia < atom_end; ia++) {
                if (ia % omp_get_num_threads() == tid) {
                    int ialoc  = ia - atom_begin;
                    auto& atom = ctx.unit_cell().atom(ia);
                    auto& type = atom.type();

                    /* wrapper for matching coefficients for a given atom */
                    mdarray<double_complex, 2> alm_tmp;
                    mdarray<double_complex, 2> halm_tmp;
                    switch (ctx.processing_unit()) {
                        case device_t::CPU: {
                            alm_tmp = mdarray<double_complex, 2>(alm_block.at(memory_t::host, 0, offsets_aw[ialoc]),
                                                                 ngv, type.mt_aw_basis_size());
                            if (hphi__ != nullptr) {
                                halm_tmp = mdarray<double_complex, 2>(
                                    halm_block.at(memory_t::host, 0, offsets_aw[ialoc]), ngv, type.mt_aw_basis_size());
                            }
                            break;
                        }
                        case device_t::GPU: {
                            alm_tmp = mdarray<double_complex, 2>(alm_block.at(memory_t::host, 0, offsets_aw[ialoc]),
                                                                 alm_block.at(memory_t::device, 0, offsets_aw[ialoc]),
                                                                 ngv, type.mt_aw_basis_size());
                            if (hphi__ != nullptr) {
                                halm_tmp =
                                    mdarray<double_complex, 2>(halm_block.at(memory_t::host, 0, offsets_aw[ialoc]),
                                                               halm_block.at(memory_t::device, 0, offsets_aw[ialoc]),
                                                               ngv, type.mt_aw_basis_size());
                            }
                            break;
                        }
                    }

                    /* generate LAPW matching coefficients on the CPU */
                    kp().alm_coeffs_loc().generate<true>(atom, alm_tmp);
                    ///* conjugate alm */
                    //for (int xi = 0; xi < type.mt_aw_basis_size(); xi++) {
                    //    for (int igk = 0; igk < ngv; igk++) {
                    //        alm_tmp(igk, xi) = std::conj(alm_tmp(igk, xi));
                    //    }
                    //}
                    if (ctx.processing_unit() == device_t::GPU) {
                        alm_tmp.copy_to(memory_t::device, stream_id(tid));
                    }
                    if (hphi__ != nullptr) {
                        H0_.apply_hmt_to_apw<spin_block_t::nm>(atom, ngv, alm_tmp, halm_tmp);
                        if (ctx.processing_unit() == device_t::GPU) {
                            halm_tmp.copy_to(memory_t::device, stream_id(tid));
                        }
                    }
                }
            }
            if (ctx.processing_unit() == device_t::GPU) {
                acc::sync_stream(stream_id(tid));
            }
        }
    };

    auto compute_alm_phi = [&](matrix<double_complex>& alm_phi, matrix<double_complex>& halm_phi, int num_mt_aw) {
        utils::timer t1("sirius::Hamiltonian_k::apply_fv_h_o|alm_phi");

        /* first zgemm: A(G, lm)^{T} * C(G, i) and  hA(G, lm)^{T} * C(G, i) */
        switch (ctx.processing_unit()) {
            case device_t::CPU: {
                if (ophi__ != nullptr) {
                    /* create resulting array with proper dimensions from the already allocated chunk of memory */
                    alm_phi = matrix<double_complex>(alm_phi_buf.at(memory_t::host), num_mt_aw, n__);
                    /* alm_phi(lm, i) = A(G, lm)^{T} * C(G, i), remember that Alm was conjugated */
                    linalg<device_t::CPU>::gemm(2, 0, num_mt_aw, n__, ngv, alm_block.at(memory_t::host), alm_block.ld(),
                                                phi__.pw_coeffs(0).prime().at(memory_t::host, 0, N__),
                                                phi__.pw_coeffs(0).prime().ld(), alm_phi.at(memory_t::host),
                                                alm_phi.ld());
                }
                if (hphi__ != nullptr) {
                    /* create resulting array with proper dimensions from the already allocated chunk of memory */
                    halm_phi = matrix<double_complex>(halm_phi_buf.at(memory_t::host), num_mt_aw, n__);
                    /* halm_phi(lm, i) = H_{mt}A(G, lm)^{T} * C(G, i) */
                    linalg<device_t::CPU>::gemm(2, 0, num_mt_aw, n__, ngv, halm_block.at(memory_t::host),
                                                halm_block.ld(), phi__.pw_coeffs(0).prime().at(memory_t::host, 0, N__),
                                                phi__.pw_coeffs(0).prime().ld(), halm_phi.at(memory_t::host),
                                                halm_phi.ld());
                }
                break;
            }
            case device_t::GPU: {
#if defined(__GPU)
                if (ophi__ != nullptr) {
                    /* create resulting array with proper dimensions from the already allocated chunk of memory */
                    alm_phi = matrix<double_complex>(alm_phi_buf.at(memory_t::host), alm_phi_buf.at(memory_t::device),
                                                     num_mt_aw, n__);
                    /* alm_phi(lm, i) = A(G, lm)^{T} * C(G, i) */
                    linalg<device_t::GPU>::gemm(2, 0, num_mt_aw, n__, ngv, alm_block.at(memory_t::device),
                                                alm_block.ld(), phi__.pw_coeffs(0).prime().at(memory_t::device, 0, N__),
                                                phi__.pw_coeffs(0).prime().ld(), alm_phi.at(memory_t::device),
                                                alm_phi.ld());
                    alm_phi.copy_to(memory_t::host);
                }
                if (hphi__ != nullptr) {
                    /* create resulting array with proper dimensions from the already allocated chunk of memory */
                    halm_phi = matrix<double_complex>(halm_phi_buf.at(memory_t::host),
                                                      halm_phi_buf.at(memory_t::device), num_mt_aw, n__);
                    /* halm_phi(lm, i) = H_{mt}A(G, lm)^{T} * C(G, i) */
                    linalg<device_t::GPU>::gemm(
                        2, 0, num_mt_aw, n__, ngv, halm_block.at(memory_t::device), halm_block.ld(),
                        phi__.pw_coeffs(0).prime().at(memory_t::device, 0, N__), phi__.pw_coeffs(0).prime().ld(),
                        halm_phi.at(memory_t::device), halm_phi.ld());

                    halm_phi.copy_to(memory_t::host);
                }
#endif
                break;
            }
        }

        if (hphi__ != nullptr) {
            kp().comm().allreduce(halm_phi.at(memory_t::host), num_mt_aw * n__);
            if (ctx.processing_unit() == device_t::GPU) {
                halm_phi.copy_to(memory_t::device);
            }
        }

        if (ophi__ != nullptr) {
            kp().comm().allreduce(alm_phi.at(memory_t::host), num_mt_aw * n__);
            if (ctx.processing_unit() == device_t::GPU) {
                alm_phi.copy_to(memory_t::device);
            }
        }
    };

    auto compute_apw_apw = [&](matrix<double_complex>& alm_phi, matrix<double_complex>& halm_phi, int num_mt_aw) {
        utils::timer t1("sirius::Hamiltonian_k::apply_fv_h_o|apw-apw");
        /* second zgemm: Alm^{*} (Alm * C) */
        switch (ctx.processing_unit()) {
            case device_t::CPU: {
                if (ophi__ != nullptr) {
                    /* APW-APW contribution to overlap */
                    linalg<device_t::CPU>::gemm(
                        0, 0, ngv, n__, num_mt_aw, linalg_const<double_complex>::one(), alm_block.at(memory_t::host),
                        alm_block.ld(), alm_phi.at(memory_t::host), alm_phi.ld(), linalg_const<double_complex>::one(),
                        ophi__->pw_coeffs(0).prime().at(memory_t::host, 0, N__), ophi__->pw_coeffs(0).prime().ld());
                }
                if (hphi__ != nullptr) {
                    /* APW-APW contribution to Hamiltonian */
                    linalg<device_t::CPU>::gemm(
                        0, 0, ngv, n__, num_mt_aw, linalg_const<double_complex>::one(), alm_block.at(memory_t::host),
                        alm_block.ld(), halm_phi.at(memory_t::host), halm_phi.ld(), linalg_const<double_complex>::one(),
                        hphi__->pw_coeffs(0).prime().at(memory_t::host, 0, N__), hphi__->pw_coeffs(0).prime().ld());
                }
                break;
            }
            case device_t::GPU: {
#if defined(__GPU)
                if (ophi__ != nullptr) {
                    /* APW-APW contribution to overlap */
                    linalg<device_t::GPU>::gemm(
                        0, 0, ngv, n__, num_mt_aw, &linalg_const<double_complex>::one(), alm_block.at(memory_t::device),
                        alm_block.ld(), alm_phi.at(memory_t::device), alm_phi.ld(),
                        &linalg_const<double_complex>::one(), ophi__->pw_coeffs(0).prime().at(memory_t::device, 0, N__),
                        ophi__->pw_coeffs(0).prime().ld());
                }
                if (hphi__ != nullptr) {
                    /* APW-APW contribution to Hamiltonian */
                    linalg<device_t::GPU>::gemm(
                        0, 0, ngv, n__, num_mt_aw, &linalg_const<double_complex>::one(), alm_block.at(memory_t::device),
                        alm_block.ld(), halm_phi.at(memory_t::device), halm_phi.ld(),
                        &linalg_const<double_complex>::one(), hphi__->pw_coeffs(0).prime().at(memory_t::device, 0, N__),
                        hphi__->pw_coeffs(0).prime().ld());
                }
#endif
                break;
            }
        }
    };

    auto collect_lo = [&](int atom_begin, int atom_end, std::vector<int>& offsets_lo,
                          matrix<double_complex>& phi_lo_block) {
        utils::timer t1("sirius::Hamiltonian_k::apply_fv_h_o|phi_lo");
        /* broadcast local orbital coefficients */
        for (int ia = atom_begin; ia < atom_end; ia++) {
            int ialoc        = ia - atom_begin;
            auto& atom       = ctx.unit_cell().atom(ia);
            auto& type       = atom.type();
            auto ia_location = phi__.spl_num_atoms().location(ia);

            /* lo coefficients for a given atom and all bands */
            matrix<double_complex> phi_lo_ia(type.mt_lo_basis_size(), n__);

            if (ia_location.rank == kp().comm().rank()) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < n__; i++) {
                    std::memcpy(&phi_lo_ia(0, i),
                                phi__.mt_coeffs(0).prime().at(memory_t::host,
                                                              phi__.offset_mt_coeffs(ia_location.local_index), N__ + i),
                                type.mt_lo_basis_size() * sizeof(double_complex));
                }
            }
            /* broadcast from a rank */
            kp().comm().bcast(phi_lo_ia.at(memory_t::host), static_cast<int>(phi_lo_ia.size()), ia_location.rank);
            /* wrtite into a proper position in a block */
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n__; i++) {
                std::memcpy(&phi_lo_block(offsets_lo[ialoc], i), &phi_lo_ia(0, i),
                            type.mt_lo_basis_size() * sizeof(double_complex));
            }
        } // ia

        if (ctx.processing_unit() == device_t::GPU) {
            phi_lo_block.copy_to(memory_t::device);
        }
    };

    auto compute_apw_lo = [&](int atom_begin, int atom_end, int num_mt_lo, std::vector<int>& offsets_aw,
                              std::vector<int> offsets_lo, matrix<double_complex>& phi_lo_block) {
        utils::timer t1("sirius::Hamiltonian_k::apply_fv_h_o|apw-lo");
        /* apw-lo block for hphi */
        if (hphi__ != nullptr) {
            for (int ia = atom_begin; ia < atom_end; ia++) {
                int ialoc  = ia - atom_begin;
                auto& atom = ctx.unit_cell().atom(ia);
                auto& type = atom.type();
                int naw    = type.mt_aw_basis_size();
                int nlo    = type.mt_lo_basis_size();

                matrix<double_complex> hmt(naw, nlo);
                #pragma omp parallel for schedule(static)
                for (int ilo = 0; ilo < nlo; ilo++) {
                    int xi_lo = naw + ilo;
                    /* local orbital indices */
                    int lm_lo    = type.indexb(xi_lo).lm;
                    int idxrf_lo = type.indexb(xi_lo).idxrf;
                    for (int xi = 0; xi < naw; xi++) {
                        int lm_aw    = type.indexb(xi).lm;
                        int idxrf_aw = type.indexb(xi).idxrf;
                        auto& gc     = H0_.gaunt_coefs().gaunt_vector(lm_aw, lm_lo);
                        hmt(xi, ilo) = atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_aw, idxrf_lo, gc);
                    }
                }
                switch (ctx.processing_unit()) {
                    case device_t::CPU: {
                        linalg<device_t::CPU>::gemm(
                            0, 0, ngv, nlo, naw, alm_block.at(memory_t::host, 0, offsets_aw[ialoc]), alm_block.ld(),
                            hmt.at(memory_t::host), hmt.ld(), halm_block.at(memory_t::host, 0, offsets_lo[ialoc]),
                            halm_block.ld());
                        break;
                    }
                    case device_t::GPU: {
#if defined(__GPU)
                        hmt.allocate(memory_t::device);
                        hmt.copy_to(memory_t::device);
                        linalg<device_t::GPU>::gemm(
                            0, 0, ngv, nlo, naw, alm_block.at(memory_t::device, 0, offsets_aw[ialoc]), alm_block.ld(),
                            hmt.at(memory_t::device), hmt.ld(), halm_block.at(memory_t::device, 0, offsets_lo[ialoc]),
                            halm_block.ld());
#endif
                        break;
                    }
                }
            } // ia
            switch (ctx.processing_unit()) {
                case device_t::CPU: {
                    linalg<device_t::CPU>::gemm(
                        0, 0, ngv, n__, num_mt_lo, linalg_const<double_complex>::one(), halm_block.at(memory_t::host),
                        halm_block.ld(), phi_lo_block.at(memory_t::host), phi_lo_block.ld(),
                        linalg_const<double_complex>::one(), hphi__->pw_coeffs(0).prime().at(memory_t::host, 0, N__),
                        hphi__->pw_coeffs(0).prime().ld());
                    break;
                }
                case device_t::GPU: {
#if defined(__GPU)
                    linalg<device_t::GPU>::gemm(
                        0, 0, ngv, n__, num_mt_lo, &linalg_const<double_complex>::one(),
                        halm_block.at(memory_t::device), halm_block.ld(), phi_lo_block.at(memory_t::device),
                        phi_lo_block.ld(), &linalg_const<double_complex>::one(),
                        hphi__->pw_coeffs(0).prime().at(memory_t::device, 0, N__), hphi__->pw_coeffs(0).prime().ld());
#endif
                    break;
                }
            }
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
                                atom.symmetry_class().o_radial_integral(l_lo, order_aw, order_lo);
                        } // TODO: block copy to GPU
                    }
                }
            } // ia
            switch (ctx.processing_unit()) {
                case device_t::CPU: {
                    linalg<device_t::CPU>::gemm(
                        0, 0, ngv, n__, num_mt_lo, linalg_const<double_complex>::one(), halm_block.at(memory_t::host),
                        halm_block.ld(), phi_lo_block.at(memory_t::host), phi_lo_block.ld(),
                        linalg_const<double_complex>::one(), ophi__->pw_coeffs(0).prime().at(memory_t::host, 0, N__),
                        ophi__->pw_coeffs(0).prime().ld());
                    break;
                }
                case device_t::GPU: {
#if defined(__GPU)
                    halm_block.copy_to(memory_t::device, 0, ngv * num_mt_lo);
                    linalg<device_t::GPU>::gemm(
                        0, 0, ngv, n__, num_mt_lo, &linalg_const<double_complex>::one(),
                        halm_block.at(memory_t::device), halm_block.ld(), phi_lo_block.at(memory_t::device),
                        phi_lo_block.ld(), &linalg_const<double_complex>::one(),
                        ophi__->pw_coeffs(0).prime().at(memory_t::device, 0, N__), ophi__->pw_coeffs(0).prime().ld());
#endif
                    break;
                }
            }
        }
    };

    utils::timer t2("sirius::Hamiltonian_k::apply_fv_h_o|mt");
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

        matrix<double_complex> alm_phi;
        matrix<double_complex> halm_phi;

        compute_alm_phi(alm_phi, halm_phi, num_mt_aw);

        if (!phi_is_lo__) {
            compute_apw_apw(alm_phi, halm_phi, num_mt_aw);
        }

        if (!apw_only__ && num_mt_lo) {
            /* local orbital coefficients for a block of atoms and all states */
            matrix<double_complex> phi_lo_block(num_mt_lo, n__);
            if (ctx.processing_unit() == device_t::GPU) {
                phi_lo_block.allocate(memory_t::device);
            }
            collect_lo(atom_begin, atom_end, offsets_lo, phi_lo_block);

            compute_apw_lo(atom_begin, atom_end, num_mt_lo, offsets_aw, offsets_lo, phi_lo_block);

            utils::timer t3("sirius::Hamiltonian::apply_fv_h_o|lo-lo-apw");
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
                                        atom.symmetry_class().o_radial_integral(l_lo, order_lo, order1);
                                }
                            }
                            if (hphi__ != nullptr) {
                                auto& gc = H0_.gaunt_coefs().gaunt_vector(lm_lo, lm1);
                                for (int i = 0; i < n__; i++) {
                                    hphi__->mt_coeffs(0).prime(offset_mt_coeffs + ilo, N__ + i) +=
                                        phi_lo_block(offsets_lo[ialoc] + jlo, i) *
                                        atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_lo, idxrf1, gc);
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
                                            atom.symmetry_class().o_radial_integral(l_lo, order_lo, order_aw) *
                                            alm_phi(offsets_aw[ialoc] + type.indexb_by_lm_order(lm_lo, order_aw), i);
                                    }
                                }
                            }

                            if (hphi__ != nullptr) {
                                for (int i = 0; i < n__; i++) {
                                    double_complex z(0, 0);
                                    for (int xi = 0; xi < type.mt_aw_basis_size(); xi++) {
                                        int lm_aw    = type.indexb(xi).lm;
                                        int idxrf_aw = type.indexb(xi).idxrf;
                                        auto& gc     = H0_.gaunt_coefs().gaunt_vector(lm_lo, lm_aw);
                                        z += atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_lo, idxrf_aw, gc) *
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
            t3.stop();
        }
    }
    t2.stop();
    if (ctx.processing_unit() == device_t::GPU && !apw_only__) {
        if (hphi__ != nullptr) {
            hphi__->mt_coeffs(0).copy_to(memory_t::device, N__, n__);
        }
        if (ophi__ != nullptr) {
            ophi__->mt_coeffs(0).copy_to(memory_t::device, N__, n__);
        }
    }
    if (ctx.control().print_checksum_) {
        if (hphi__) {
            auto cs1 = hphi__->checksum_pw(ctx.processing_unit(), 0, N__, n__);
            auto cs2 = hphi__->checksum(ctx.processing_unit(), 0, N__, n__);
            if (kp().comm().rank() == 0) {
                utils::print_checksum("hphi_pw", cs1);
                utils::print_checksum("hphi", cs2);
            }
        }
        if (ophi__) {
            auto cs1 = ophi__->checksum_pw(ctx.processing_unit(), 0, N__, n__);
            auto cs2 = ophi__->checksum(ctx.processing_unit(), 0, N__, n__);
            if (kp().comm().rank() == 0) {
                utils::print_checksum("ophi_pw", cs1);
                utils::print_checksum("ophi", cs2);
            }
        }
    }
}

void Hamiltonian_k::apply_b(Wave_functions& psi__, std::vector<Wave_functions>& bpsi__)
{
    PROFILE("sirius::Hamiltonian_k::apply_b");

    assert(bpsi__.size() == 2 || bpsi__.size() == 3);

    H0().local_op().apply_b(kp().spfft_transform(), 0, H0().ctx().num_fv_states(), psi__, bpsi__);
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

template void Hamiltonian_k::apply_h_s<double>(spin_range spins__, int N__, int n__, Wave_functions& phi__,
                                               Wave_functions* hphi__, Wave_functions* sphi__);
template void Hamiltonian_k::apply_h_s<double_complex>(spin_range spins__, int N__, int n__, Wave_functions& phi__,
                                                       Wave_functions* hphi__, Wave_functions* sphi__);

} // namespace sirius
