/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file augmentation_operator.cpp
 *
 *  \brief Contains implementation of sirius::Augmentation_operator class.
 */

#include "augmentation_operator.hpp"

namespace sirius {

void
Augmentation_operator::generate_pw_coeffs()
{
    if (!atom_type_.augment()) {
        return;
    }
    PROFILE("sirius::Augmentation_operator::generate_pw_coeffs");

    double fourpi_omega = fourpi / gvec_.omega();

    auto const& tp = gvec_.gvec_tp();

    /* maximum l of beta-projectors */
    int lmax_beta = atom_type_.indexr().lmax();
    int lmmax     = sf::lmmax(2 * lmax_beta);

    /* number of beta-projectors */
    int nbf = atom_type_.mt_basis_size();
    /* only half of Q_{xi,xi'}(G) matrix is stored */
    int nqlm = nbf * (nbf + 1) / 2;
    /* local number of G-vectors */
    int gvec_count = gvec_.count();

    /* Info:
     *   After some tests, the current GPU implementation of generating aug. operator turns out to be slower than CPU.
     *   The reason is probaly the memory access pattern of G, lm and, idxrf indices.
     *   The current decision is to compute aug. operator on CPU once during the initialization and
     *   then copy the chunks of Q(G) to GPU when computing D-operator and augment charge density.
     */
    switch (atom_type_.parameters().processing_unit()) {
        case device_t::CPU:
        case device_t::GPU: {
            /* Gaunt coefficients of three real spherical harmonics */
            Gaunt_coefficients<double> gaunt_coefs(lmax_beta, 2 * lmax_beta, lmax_beta, SHT::gaunt_rrr);
            #pragma omp parallel for
            for (int igloc = 0; igloc < gvec_count; igloc++) {
                std::vector<double> rlm(lmmax);
                sf::spherical_harmonics(2 * lmax_beta, tp(igloc, 0), tp(igloc, 1), rlm.data());
                std::vector<std::complex<double>> v(lmmax);
                for (int idx12 = 0; idx12 < nqlm; idx12++) {
                    int lm1     = idx_(0, idx12);
                    int lm2     = idx_(1, idx12);
                    int idxrf12 = idx_(2, idx12);
                    for (int lm3 = 0; lm3 < lmmax; lm3++) {
                        v[lm3] = std::conj(zilm_[lm3]) * rlm[lm3] *
                                 ri_values_(idxrf12, l_by_lm_[lm3], gvec_.gvec_shell_idx_local(igloc));
                    }
                    std::complex<double> z      = fourpi_omega * gaunt_coefs.sum_L3_gaunt(lm2, lm1, &v[0]);
                    q_pw_(idx12, 2 * igloc)     = z.real();
                    q_pw_(idx12, 2 * igloc + 1) = z.imag();
                }
            }
            break;
        }
            //        case device_t::GPU: {
            // #if defined(SIRIUS_GPU)
            //            auto spl_ngv_loc = utils::split_in_blocks(gvec_count,
            //            atom_type_.parameters().cfg().control().gvec_chunk_size()); auto& mpd =
            //            sddk::get_memory_pool(sddk::memory_t::device);
            //            /* allocate buffer for Rlm on GPUs */
            //            mdarray<double, 2> gvec_rlm(lmmax, spl_ngv_loc[0], mpd, "gvec_rlm");
            //            /* allocate buffer for Q(G) on GPUs */
            //            mdarray<double, 2> qpw(nqlm, 2 * spl_ngv_loc[0], mpd, "qpw");
            //
            //            int g_begin{0};
            //            /* loop over blocks of G-vectors */
            //            for (auto ng : spl_ngv_loc) {
            //                /* generate Rlm spherical harmonics */
            //                spherical_harmonics_rlm_gpu(2 * lmax_beta, ng, tp.at(sddk::memory_t::device, g_begin, 0),
            //                        tp.at(sddk::memory_t::device, g_begin, 1), gvec_rlm.at(sddk::memory_t::device),
            //                        gvec_rlm.ld());
            //                this->generate_pw_coeffs_chunk_gpu(g_begin, ng, gvec_rlm.at(sddk::memory_t::device),
            //                gvec_rlm.ld(), qpw); acc::copyout(q_pw_.at(sddk::memory_t::host, 0, 2 * g_begin),
            //                qpw.at(sddk::memory_t::device), 2 * ng * nqlm); g_begin += ng;
            //            }
            // #endif
            //            break;
            //        }
    }

    q_mtrx_ = mdarray<double, 2>({nbf, nbf});
    q_mtrx_.zero();

    if (gvec_.comm().rank() == 0) {
        for (int xi2 = 0; xi2 < nbf; xi2++) {
            for (int xi1 = 0; xi1 <= xi2; xi1++) {
                /* packed orbital index */
                int idx12         = packed_index(xi1, xi2);
                q_mtrx_(xi1, xi2) = q_mtrx_(xi2, xi1) = gvec_.omega() * q_pw_(idx12, 0);
            }
        }
    }
    /* broadcast from rank#0 */
    gvec_.comm().bcast(&q_mtrx_(0, 0), nbf * nbf, 0);

    if (env::print_checksum()) {
        auto cs  = q_pw_.checksum();
        auto cs1 = q_mtrx_.checksum();
        gvec_.comm().allreduce(&cs, 1);
        if (gvec_.comm().rank() == 0) {
            print_checksum("q_pw", cs, std::cout);
            print_checksum("q_mtrx", cs1, std::cout);
        }
    }
}

void
Augmentation_operator::generate_pw_coeffs_gvec_deriv(int nu__)
{
    if (!atom_type_.augment()) {
        return;
    }
    PROFILE("sirius::Augmentation_operator::generate_pw_coeffs_gvec_deriv");

    auto const& tp = gvec_.gvec_tp();

    /* maximum l of beta-projectors */
    int lmax_beta = atom_type_.indexr().lmax();
    int lmmax     = sf::lmmax(2 * lmax_beta);

    /* number of beta-projectors */
    int nbf = atom_type_.mt_basis_size();
    /* local number of G-vectors */
    int gvec_count = gvec_.count();

    switch (atom_type_.parameters().processing_unit()) {
        case device_t::GPU:
        case device_t::CPU: {
            /* Gaunt coefficients of three real spherical harmonics */
            Gaunt_coefficients<double> gaunt_coefs(lmax_beta, 2 * lmax_beta, lmax_beta, SHT::gaunt_rrr);
            #pragma omp parallel for
            for (int igloc = 0; igloc < gvec_count; igloc++) {
                /* index of the G-vector shell */
                int igsh = gvec_.gvec_shell_idx_local(igloc);

                auto gvc      = gvec_.gvec_cart(gvec_index_t::local(igloc));
                double gvc_nu = gvc[nu__];

                std::vector<double> rlm(lmmax);
                mdarray<double, 2> rlm_dq({lmmax, 3});
                sf::spherical_harmonics(2 * lmax_beta, tp(igloc, 0), tp(igloc, 1), rlm.data());
                const bool divide_by_r{false};
                sf::dRlm_dr(2 * lmax_beta, gvc, rlm_dq, divide_by_r);

                std::vector<std::complex<double>> v(lmmax);
                for (int idx12 = 0; idx12 < nbf * (nbf + 1) / 2; idx12++) {
                    int lm1     = idx_(0, idx12);
                    int lm2     = idx_(1, idx12);
                    int idxrf12 = idx_(2, idx12);
                    for (int lm3 = 0; lm3 < lmmax; lm3++) {
                        int l  = l_by_lm_[lm3];
                        v[lm3] = std::conj(zilm_[lm3]) * (rlm_dq(lm3, nu__) * ri_values_(idxrf12, l, igsh) +
                                                          rlm[lm3] * ri_dq_values_(idxrf12, l, igsh) * gvc_nu);
                    }
                    std::complex<double> z      = fourpi * gaunt_coefs.sum_L3_gaunt(lm2, lm1, &v[0]);
                    q_pw_(idx12, 2 * igloc)     = z.real();
                    q_pw_(idx12, 2 * igloc + 1) = z.imag();
                }
            }
            break;
        }
    }
}

} // namespace sirius
