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

/** \file augmentation_operator.cpp
 *
 *  \brief Contains implementation of sirius::Augmentation_operator class.
 */

#include "augmentation_operator.hpp"

namespace sirius {

void Augmentation_operator::generate_pw_coeffs()
{
    if (!atom_type_.augment()) {
        return;
    }
    PROFILE("sirius::Augmentation_operator::generate_pw_coeffs");

    double fourpi_omega = fourpi / gvec_.omega();

    auto const& tp = gvec_.gvec_tp();

    /* maximum l of beta-projectors */
    int lmax_beta = atom_type_.indexr().lmax();
    int lmmax     = utils::lmmax(2 * lmax_beta);

    /* number of beta-projectors */
    int nbf = atom_type_.mt_basis_size();
    /* only half of Q_{xi,xi'}(G) matrix is stored */
    int nqlm = nbf * (nbf + 1) / 2;
    /* local number of G-vectors */
    int gvec_count = gvec_.count();
    /* array of plane-wave coefficients */
    q_pw_ = sddk::mdarray<double, 2>(nqlm, 2 * gvec_count, sddk::get_memory_pool(sddk::memory_t::host), "q_pw_");

    switch (atom_type_.parameters().processing_unit()) {
        case sddk::device_t::CPU: {
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
                    std::complex<double> z = fourpi_omega * gaunt_coefs.sum_L3_gaunt(lm2, lm1, &v[0]);
                    q_pw_(idx12, 2 * igloc)     = z.real();
                    q_pw_(idx12, 2 * igloc + 1) = z.imag();
                }
            }
            break;
        }
        case sddk::device_t::GPU: {
#if defined(SIRIUS_GPU)
            auto spl_ngv_loc = utils::split_in_blocks(gvec_count, 10000);
            auto& mpd = sddk::get_memory_pool(sddk::memory_t::device);
            /* allocate buffer for Rlm on GPUs */
            sddk::mdarray<double, 2> gvec_rlm(lmmax, spl_ngv_loc[0], mpd, "gvec_rlm");
            /* allocate buffer for Q(G) on GPUs */
            sddk::mdarray<double, 2> qpw(nqlm, 2 * spl_ngv_loc[0], mpd, "qpw");

            int g_begin{0};
            /* loop over blocks of G-vectors */
            for (auto ng : spl_ngv_loc) {
                /* generate Rlm spherical harmonics */
                spherical_harmonics_rlm_gpu(2 * lmax_beta, ng, tp.at(sddk::memory_t::device, g_begin, 0),
                        tp.at(sddk::memory_t::device, g_begin, 1), gvec_rlm.at(sddk::memory_t::device), gvec_rlm.ld());
                this->generate_pw_coeffs_chunk_gpu(g_begin, ng, gvec_rlm, qpw);
                acc::copyout(q_pw_.at(sddk::memory_t::host, 0, 2 * g_begin), qpw.at(sddk::memory_t::device), 2 * ng * nqlm);
                g_begin += ng;
            }
#endif
            break;
        }
    }

    sym_weight_ = sddk::mdarray<double, 1>(nbf * (nbf + 1) / 2, sddk::get_memory_pool(sddk::memory_t::host), "sym_weight_");
    for (int xi2 = 0; xi2 < nbf; xi2++) {
        for (int xi1 = 0; xi1 <= xi2; xi1++) {
            /* packed orbital index */
            int idx12          = utils::packed_index(xi1, xi2);
            sym_weight_(idx12) = (xi1 == xi2) ? 1 : 2;
        }
    }

    q_mtrx_ = sddk::mdarray<double, 2>(nbf, nbf);
    q_mtrx_.zero();

    if (gvec_.comm().rank() == 0) {
        for (int xi2 = 0; xi2 < nbf; xi2++) {
            for (int xi1 = 0; xi1 <= xi2; xi1++) {
                /* packed orbital index */
                int idx12         = utils::packed_index(xi1, xi2);
                q_mtrx_(xi1, xi2) = q_mtrx_(xi2, xi1) = gvec_.omega() * q_pw_(idx12, 0);
            }
        }
    }
    /* broadcast from rank#0 */
    gvec_.comm().bcast(&q_mtrx_(0, 0), nbf * nbf, 0);

    if (atom_type_.parameters().cfg().control().print_checksum()) {
        auto cs = q_pw_.checksum();
        auto cs1 = q_mtrx_.checksum();
        gvec_.comm().allreduce(&cs, 1);
        if (gvec_.comm().rank() == 0) {
            utils::print_checksum("q_pw", cs, std::cout);
            utils::print_checksum("q_mtrx", cs1, std::cout);
        }
    }
}

Augmentation_operator_gvec_deriv::Augmentation_operator_gvec_deriv(Simulation_parameters const& param__, int lmax__,
    sddk::Gvec const& gvec__)
    : gvec_(gvec__)
{
    PROFILE("sirius::Augmentation_operator_gvec_deriv");

    int lmmax = utils::lmmax(2 * lmax__);

    auto& tp = gvec__.gvec_tp();

    /* Gaunt coefficients of three real spherical harmonics */
    gaunt_coefs_ = std::unique_ptr<Gaunt_coefficients<double>>(
        new Gaunt_coefficients<double>(lmax__, 2 * lmax__, lmax__, SHT::gaunt_rrr));

    /* split G-vectors between ranks */
    int gvec_count = gvec__.count();

    rlm_g_  = sddk::mdarray<double, 2>(lmmax, gvec_count);
    rlm_dg_ = sddk::mdarray<double, 3>(lmmax, 3, gvec_count);

    /* array of real spherical harmonics and derivatives for each G-vector */
    #pragma omp parallel for schedule(static)
    for (int igloc = 0; igloc < gvec_count; igloc++) {
        /* compute Rlm */
        sf::spherical_harmonics(2 * lmax__, tp(igloc, 0), tp(igloc, 1), &rlm_g_(0, igloc));
        /* compute dRlm/dG */
        sddk::mdarray<double, 2> tmp(&rlm_dg_(0, 0, igloc), lmmax, 3);
        auto gv = gvec__.gvec_cart<sddk::index_domain_t::local>(igloc);
        sf::dRlm_dr(2 * lmax__, gv, tmp, false);
    }
    switch (param__.processing_unit()) {
        case sddk::device_t::CPU: {
            break;
        }
        case sddk::device_t::GPU: {
            rlm_g_.allocate(get_memory_pool(sddk::memory_t::device)).copy_to(sddk::memory_t::device);
            rlm_dg_.allocate(get_memory_pool(sddk::memory_t::device)).copy_to(sddk::memory_t::device);
        }
    }
}

void Augmentation_operator_gvec_deriv::prepare(Atom_type const& atom_type__,
    Radial_integrals_aug<false> const& ri__, Radial_integrals_aug<true> const& ri_dq__)
{
    PROFILE("sirius::Augmentation_operator_gvec_deriv::prepare");

    int lmax_beta = atom_type__.lmax_beta();

    /* number of beta- radial functions */
    int nbrf = atom_type__.mt_radial_basis_size();

    auto& mp = get_memory_pool(sddk::memory_t::host);

    ri_values_ = sddk::mdarray<double, 3>(2 * lmax_beta + 1, nbrf * (nbrf + 1) / 2, gvec_.num_gvec_shells_local(), mp);
    ri_dg_values_ = sddk::mdarray<double, 3>(2 * lmax_beta + 1, nbrf * (nbrf + 1) / 2, gvec_.num_gvec_shells_local(),
        mp);
    #pragma omp parallel for
    for (int j = 0; j < gvec_.num_gvec_shells_local(); j++) {
        auto ri = ri__.values(atom_type__.id(), gvec_.gvec_shell_len_local(j));
        auto ri_dg = ri_dq__.values(atom_type__.id(), gvec_.gvec_shell_len_local(j));
        for (int l = 0; l <= 2 * lmax_beta; l++) {
            for (int i = 0; i < nbrf * (nbrf + 1) / 2; i++) {
                ri_values_(l, i, j) = ri(i, l);
                ri_dg_values_(l, i, j) = ri_dg(i, l);
            }
        }
    }

    /* number of beta-projectors */
    int nbf = atom_type__.mt_basis_size();

    int idxmax = nbf * (nbf + 1) / 2;

    /* flatten the indices */
    idx_ = sddk::mdarray<int, 2>(3, idxmax, mp);
    for (int xi2 = 0; xi2 < nbf; xi2++) {
        int lm2    = atom_type__.indexb(xi2).lm;
        int idxrf2 = atom_type__.indexb(xi2).idxrf;

        for (int xi1 = 0; xi1 <= xi2; xi1++) {
            int lm1    = atom_type__.indexb(xi1).lm;
            int idxrf1 = atom_type__.indexb(xi1).idxrf;

            /* packed orbital index */
            int idx12 = utils::packed_index(xi1, xi2);
            /* packed radial-function index */
            int idxrf12 = utils::packed_index(idxrf1, idxrf2);

            idx_(0, idx12) = lm1;
            idx_(1, idx12) = lm2;
            idx_(2, idx12) = idxrf12;
        }
    }

    int gvec_count  = gvec_.count();

    gvec_shell_ = sddk::mdarray<int, 1>(gvec_count, get_memory_pool(sddk::memory_t::host));
    gvec_cart_ = sddk::mdarray<double, 2>(3, gvec_count, get_memory_pool(sddk::memory_t::host));
    for (int igloc = 0; igloc < gvec_count; igloc++) {
        auto gvc = gvec_.gvec_cart<sddk::index_domain_t::local>(igloc);
        gvec_shell_(igloc) = gvec_.gvec_shell_idx_local(igloc);
        for (int x: {0, 1, 2}) {
            gvec_cart_(x, igloc) = gvc[x];
        }
    }

    sym_weight_ = sddk::mdarray<double, 1>(nbf * (nbf + 1) / 2, mp, "sym_weight_");
    for (int xi2 = 0; xi2 < nbf; xi2++) {
        for (int xi1 = 0; xi1 <= xi2; xi1++) {
            /* packed orbital index */
            int idx12          = xi2 * (xi2 + 1) / 2 + xi1;
            sym_weight_(idx12) = (xi1 == xi2) ? 1 : 2;
        }
    }

    switch (atom_type__.parameters().processing_unit()) {
        case sddk::device_t::CPU: {
            /* array of plane-wave coefficients */
            q_pw_ = sddk::mdarray<double, 2>(nbf * (nbf + 1) / 2, 2 * gvec_count, mp, "q_pw_dg_");
            break;
        }
        case sddk::device_t::GPU: {
            auto& mpd = get_memory_pool(sddk::memory_t::device);
            ri_values_.allocate(mpd).copy_to(sddk::memory_t::device);
            ri_dg_values_.allocate(mpd).copy_to(sddk::memory_t::device);
            idx_.allocate(mpd).copy_to(sddk::memory_t::device);
            gvec_shell_.allocate(mpd).copy_to(sddk::memory_t::device);
            gvec_cart_.allocate(mpd).copy_to(sddk::memory_t::device);
            /* array of plane-wave coefficients */
            q_pw_ = sddk::mdarray<double, 2>(nbf * (nbf + 1) / 2, 2 * gvec_count, mpd, "q_pw_dg_");
            break;
        }
    }
}

void Augmentation_operator_gvec_deriv::generate_pw_coeffs(Atom_type const& atom_type__, int nu__)
{
    PROFILE("sirius::Augmentation_operator_gvec_deriv::generate_pw_coeffs");

    /* maximum l of beta-projectors */
    int lmax_beta = atom_type__.indexr().lmax();

    int lmax_q = 2 * lmax_beta;

    int lmmax = utils::lmmax(lmax_q);

    auto l_by_lm = utils::l_by_lm(lmax_q);

    sddk::mdarray<std::complex<double>, 1> zilm(lmmax);
    for (int l = 0, lm = 0; l <= lmax_q; l++) {
        for (int m = -l; m <= l; m++, lm++) {
            zilm[lm] = std::pow(std::complex<double>(0, 1), l);
        }
    }

    /* number of beta-projectors */
    int nbf = atom_type__.mt_basis_size();

    /* split G-vectors between ranks */
    int gvec_count  = gvec_.count();

    switch (atom_type__.parameters().processing_unit()) {
        case sddk::device_t::CPU: {
            #pragma omp parallel for schedule(static)
            for (int igloc = 0; igloc < gvec_count; igloc++) {
                /* index of the G-vector shell */
                int igsh = gvec_.gvec_shell_idx_local(igloc);
                std::vector<std::complex<double>> v(lmmax);
                double gvc_nu = gvec_cart_(nu__, igloc);
                for (int idx12 = 0; idx12 < nbf * (nbf + 1) / 2; idx12++) {
                    int lm1     = idx_(0, idx12);
                    int lm2     = idx_(1, idx12);
                    int idxrf12 = idx_(2, idx12);
                    for (int lm3 = 0; lm3 < lmmax; lm3++) {
                        int l = l_by_lm[lm3];
                        v[lm3] = std::conj(zilm[lm3]) *
                            (rlm_dg_(lm3, nu__, igloc) * ri_values_(l, idxrf12, igsh) +
                             rlm_g_(lm3, igloc) * ri_dg_values_(l, idxrf12, igsh) * gvc_nu);
                    }
                    std::complex<double> z = fourpi * gaunt_coefs_->sum_L3_gaunt(lm2, lm1, &v[0]);
                    q_pw_(idx12, 2 * igloc)     = z.real();
                    q_pw_(idx12, 2 * igloc + 1) = z.imag();
                }
            }
            break;
        }
        case sddk::device_t::GPU: {
            auto& mpd = get_memory_pool(sddk::memory_t::device);

            auto gc = gaunt_coefs_->get_full_set_L3();
            gc.allocate(mpd).copy_to(sddk::memory_t::device);

#if defined(SIRIUS_GPU)
            aug_op_pw_coeffs_deriv_gpu(gvec_count, gvec_shell_.at(sddk::memory_t::device), gvec_cart_.at(sddk::memory_t::device),
                idx_.at(sddk::memory_t::device), static_cast<int>(idx_.size(1)),
                gc.at(sddk::memory_t::device), static_cast<int>(gc.size(0)), static_cast<int>(gc.size(1)),
                rlm_g_.at(sddk::memory_t::device), rlm_dg_.at(sddk::memory_t::device), static_cast<int>(rlm_g_.size(0)),
                ri_values_.at(sddk::memory_t::device), ri_dg_values_.at(sddk::memory_t::device), static_cast<int>(ri_values_.size(0)),
                static_cast<int>(ri_values_.size(1)), q_pw_.at(sddk::memory_t::device), static_cast<int>(q_pw_.size(0)),
                fourpi, nu__, lmax_q);
#endif
            break;
        }
    }
}

} // namespace sirius
