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

/** \file augmentation_operator.hpp
 *
 *  \brief Contains implementation of sirius::Augmentation_operator class.
 */

#ifndef __AUGMENTATION_OPERATOR_HPP__
#define __AUGMENTATION_OPERATOR_HPP__

#include "radial/radial_integrals.hpp"
#include "fft/gvec.hpp"

#if defined(SIRIUS_GPU)
extern "C" {

void
aug_op_pw_coeffs_gpu(int ngvec__, int const* gvec_shell__, int const* idx__, int idxmax__,
        std::complex<double> const* zilm__, int const* l_by_lm__, int lmmax__, double const* gc__, int ld0__,
        int ld1__, double const* gvec_rlm__, int ld2__, double const* ri_values__, int ld3__, int ld4__,
        double* q_pw__, int ld5__, double fourpi_omega__);

void
aug_op_pw_coeffs_deriv_gpu(int ngvec__, int const* gvec_shell__, double const* gvec_cart__, int const* idx__,
        int idxmax__, double const* gc__, int ld0__, int ld1__, double const* rlm__, double const* rlm_dg__, int ld2__,
        double const* ri_values__, double const* ri_dg_values__, int ld3__, int ld4__, double* q_pw__, int ld5__,
        double fourpi__, int nu__, int lmax_q__);

void
spherical_harmonics_rlm_gpu(int lmax__, int ntp__, double const* theta__, double const* phi__, double* rlm__, int ld__);

}
#endif


namespace sirius {

//class Augmentation_operator_new
//{
//  private:
//    Unit_cell const& unit_cell_;
//
//    sddk::Gvec const& gvec_;
//
//    sddk::mdarray<int, 1> l_by_lm_;
//
//    sddk::mdarray<std::complex<double>, 1> zilm_;
//
//    std::vector<sddk::mdarray<double, 3>> gaunt_coefs_;
//
//    std::vector<sddk::mdarray<int, 2>> idx_;
//
//    std::vector<sddk::mdarray<double, 3>> ri_values_;
//
//    sddk::mdarray<int, 1> gvec_shell_;
//
//    std::vector<sddk::mdarray<double, 2>> q_pw_;
//    std::vector<sddk::mdarray<double, 1>> sym_weight_;
//
//    void init(Radial_integrals_aug<false> const& radial_integrals__)
//    {
//        auto lmax_beta = unit_cell_.lmax();
//        auto lmax      = 2 * lmax_beta;
//        auto lmmax     = utils::lmmax(lmax);
//
//        /* local number of G-vectors */
//        auto gvec_count = gvec_.count();
//
//        /* compute l of lm index */
//        auto l_by_lm = utils::l_by_lm(lmax);
//        l_by_lm_ = sddk::mdarray<int, 1>(lmmax);
//        std::copy(l_by_lm.begin(), l_by_lm.end(), &l_by_lm_[0]);
//
//        /* compute i^l array */
//        zilm_ = sddk::mdarray<std::complex<double>, 1>(lmmax);
//        for (int l = 0, lm = 0; l <= lmax; l++) {
//            for (int m = -l; m <= l; m++, lm++) {
//                zilm_[lm] = std::pow(std::complex<double>(0, 1), l);
//            }
//        }
//
//        /* number of atom types */
//        auto nat = unit_cell_.num_atom_types();
//
//        gaunt_coefs_.resize(nat);
//        idx_.resize(nat);
//        ri_values_.resize(nat);
//        sym_weight_.resize(nat);
//        q_pw_.resize(nat);
//        for (int iat = 0; iat < nat; iat++) {
//            auto& atom_type = unit_cell_.atom_type(iat);
//            if (!atom_type.augment() || atom_type.num_atoms() == 0) {
//                continue;
//            }
//            auto lmax_t = atom_type.indexr().lmax();
//            /* Gaunt coefficients of three real spherical harmonics */
//            gaunt_coefs_[iat] = Gaunt_coefficients<double>(lmax_t, 2 * lmax_t, lmax_t, SHT::gaunt_rrr).get_full_set_L3();
//
//            /* number of beta-projectors */
//            auto nbf = atom_type.mt_basis_size();
//            /* number of beta-projector radial functions */
//            auto nbrf = atom_type.mt_radial_basis_size();
//
//            auto idxmax = nbf * (nbf + 1) / 2;
//            idx_[iat] = sddk::mdarray<int, 2>(3, idxmax);
//
//            for (int xi2 = 0; xi2 < nbf; xi2++) {
//                int lm2    = atom_type.indexb(xi2).lm;
//                int idxrf2 = atom_type.indexb(xi2).idxrf;
//
//                for (int xi1 = 0; xi1 <= xi2; xi1++) {
//                    int lm1    = atom_type.indexb(xi1).lm;
//                    int idxrf1 = atom_type.indexb(xi1).idxrf;
//
//                    /* packed orbital index */
//                    int idx12 = utils::packed_index(xi1, xi2);
//                    /* packed radial-function index */
//                    int idxrf12 = utils::packed_index(idxrf1, idxrf2);
//
//                    idx_[iat](0, idx12) = lm1;
//                    idx_[iat](1, idx12) = lm2;
//                    idx_[iat](2, idx12) = idxrf12;
//                }
//            }
//
//            sym_weight_[iat] = sddk::mdarray<double, 1>(idxmax);
//            for (int xi2 = 0; xi2 < nbf; xi2++) {
//                for (int xi1 = 0; xi1 <= xi2; xi1++) {
//                    /* packed orbital index */
//                    int idx12          = utils::packed_index(xi1, xi2);
//                    sym_weight_[iat](idx12) = (xi1 == xi2) ? 1 : 2;
//                }
//            }
//
//            ri_values_[iat] = sddk::mdarray<double, 3>(nbrf * (nbrf + 1) / 2, 2 * lmax_t + 1, gvec_.num_gvec_shells_local());
//            #pragma omp parallel for
//            for (int j = 0; j < gvec_.num_gvec_shells_local(); j++) {
//                auto ri = radial_integrals__.values(iat, gvec_.gvec_shell_len_local(j));
//                for (int l = 0; l <= lmax_t; l++) {
//                    for (int i = 0; i < nbrf * (nbrf + 1) / 2; i++) {
//                        ri_values_[iat](i, l, j) = ri(i, l);
//                    }
//                }
//            }
//
//            q_pw_[iat] = sddk::mdarray<double, 2>(idxmax, 2 * gvec_count, sddk::get_memory_pool(sddk::memory_t::host), "q_pw_");
//
//            if (unit_cell_.parameters().processing_unit() == sddk::device_t::GPU) {
//                auto& mpd = sddk::get_memory_pool(sddk::memory_t::device);
//                gaunt_coefs_[iat].allocate(mpd).copy_to(sddk::memory_t::device);
//                idx_[iat].allocate(mpd).copy_to(sddk::memory_t::device);
//                ri_values_[iat].allocate(mpd).copy_to(sddk::memory_t::device);
//                sym_weight_[iat].allocate(mpd).copy_to(sddk::memory_t::device);
//            }
//        } // iat
//
//        gvec_shell_ = sddk::mdarray<int, 1>(gvec_count);
//        for (int igloc = 0; igloc < gvec_count; igloc++) {
//            gvec_shell_(igloc) = gvec_.gvec_shell_idx_local(igloc);
//        }
//
//        if (unit_cell_.parameters().processing_unit() == sddk::device_t::GPU) {
//            auto& mpd = sddk::get_memory_pool(sddk::memory_t::device);
//            l_by_lm_.allocate(mpd).copy_to(sddk::memory_t::device);
//            zilm_.allocate(mpd).copy_to(sddk::memory_t::device);
//            gvec_shell_.allocate(mpd).copy_to(sddk::memory_t::device);
//        }
//    }
//
//    void generate_pw_coeffs_chunk_gpu(int iat__, int g_begin__, int ng__, sddk::mdarray<double, 2> const& gvec_rlm__,
//            sddk::mdarray<double, 2>& qpw__) const
//    {
//#if defined(SIRIUS_GPU)
//        double fourpi_omega = fourpi / unit_cell_.omega();
//
//        /* maximum l of beta-projectors */
//        int lmax_beta = unit_cell_.atom_type(iat__).indexr().lmax();
//        int lmmax     = utils::lmmax(2 * lmax_beta);
//        /* number of beta-projectors */
//        int nbf = unit_cell_.atom_type(iat__).mt_basis_size();
//        /* only half of Q_{xi,xi'}(G) matrix is stored */
//        int nqlm = nbf * (nbf + 1) / 2;
//        /* generate Q(G) */
//        aug_op_pw_coeffs_gpu(ng__, gvec_shell_.at(sddk::memory_t::device, g_begin__),
//                idx_[iat__].at(sddk::memory_t::device), nqlm, zilm_.at(sddk::memory_t::device),
//                l_by_lm_.at(sddk::memory_t::device), lmmax, gaunt_coefs_[iat__].at(sddk::memory_t::device),
//                static_cast<int>(gaunt_coefs_[iat__].size(0)), static_cast<int>(gaunt_coefs_[iat__].size(1)),
//                gvec_rlm__.at(sddk::memory_t::device), static_cast<int>(gvec_rlm__.size(0)),
//                ri_values_[iat__].at(sddk::memory_t::device), static_cast<int>(ri_values_[iat__].size(0)),
//                static_cast<int>(ri_values_[iat__].size(1)), qpw__.at(sddk::memory_t::device),
//                static_cast<int>(qpw__.size(0)), fourpi_omega);
//#endif
//    }
//
//
//    void generate_pw_coeffs()
//    {
//        auto fourpi_omega = fourpi / unit_cell_.omega();
//
//        auto const& tp = gvec_.gvec_tp();
//
//        /* local number of G-vectors */
//        auto gvec_count = gvec_.count();
//
//        /* number of atom types */
//        auto nat = unit_cell_.num_atom_types();
//
//        for (int iat = 0; iat < nat; iat++) {
//            auto& atom_type = unit_cell_.atom_type(iat);
//            if (!atom_type.augment() || atom_type.num_atoms() == 0) {
//                continue;
//            }
//
//            auto lmax_t = atom_type.indexr().lmax();
//            auto lmmax  = utils::lmmax(2 * lmax_t);
//            /* number of beta-projectors */
//            auto nbf = atom_type.mt_basis_size();
//            /* only half of Q_{xi,xi'}(G) matrix is stored */
//            auto nqlm = nbf * (nbf + 1) / 2;
//
//            switch (unit_cell_.parameters().processing_unit()) {
//                case sddk::device_t::CPU: {
//                    /* Gaunt coefficients of three real spherical harmonics */
//                    Gaunt_coefficients<double> gaunt_coefs(lmax_t, 2 * lmax_t, lmax_t, SHT::gaunt_rrr);
//                    #pragma omp parallel for
//                    for (int igloc = 0; igloc < gvec_count; igloc++) {
//                        std::vector<double> rlm(lmmax);
//                        sf::spherical_harmonics(2 * lmax_t, tp(igloc, 0), tp(igloc, 1), rlm.data());
//                        std::vector<std::complex<double>> v(lmmax);
//                        for (int idx12 = 0; idx12 < nqlm; idx12++) {
//                            int lm1     = idx_[iat](0, idx12);
//                            int lm2     = idx_[iat](1, idx12);
//                            int idxrf12 = idx_[iat](2, idx12);
//                            for (int lm3 = 0; lm3 < lmmax; lm3++) {
//                                v[lm3] = std::conj(zilm_[lm3]) * rlm[lm3] *
//                                    ri_values_[iat](idxrf12, l_by_lm_[lm3], gvec_.gvec_shell_idx_local(igloc));
//                            }
//                            std::complex<double> z = fourpi_omega * gaunt_coefs.sum_L3_gaunt(lm2, lm1, &v[0]);
//                            q_pw_[iat](idx12, 2 * igloc)     = z.real();
//                            q_pw_[iat](idx12, 2 * igloc + 1) = z.imag();
//                        }
//                    }
//                    break;
//                }
//                case sddk::device_t::GPU: {
//#if defined(SIRIUS_GPU)
//                    auto spl_ngv_loc = utils::split_in_blocks(gvec_count, 10000);
//                    auto& mpd = sddk::get_memory_pool(sddk::memory_t::device);
//                    /* allocate buffer for Rlm on GPUs */
//                    sddk::mdarray<double, 2> gvec_rlm(lmmax, spl_ngv_loc[0], mpd, "gvec_rlm");
//                    /* allocate buffer for Q(G) on GPUs */
//                    sddk::mdarray<double, 2> qpw(nqlm, 2 * spl_ngv_loc[0], mpd, "qpw");
//
//                    int g_begin{0};
//                    /* loop over blocks of G-vectors */
//                    for (auto ng : spl_ngv_loc) {
//                        /* generate Rlm spherical harmonics */
//                        spherical_harmonics_rlm_gpu(2 * lmax_t, ng, tp.at(sddk::memory_t::device, g_begin, 0),
//                                tp.at(sddk::memory_t::device, g_begin, 1), gvec_rlm.at(sddk::memory_t::device),
//                                gvec_rlm.ld());
//                        this->generate_pw_coeffs_chunk_gpu(iat, g_begin, ng, gvec_rlm, qpw);
//                        acc::copyout(q_pw_[iat].at(sddk::memory_t::host, 0, 2 * g_begin),
//                                qpw.at(sddk::memory_t::device), 2 * ng * nqlm);
//                        g_begin += ng;
//                    }
//#endif
//                    break;
//                }
//            }
//
//            //q_mtrx_ = sddk::mdarray<double, 2>(nbf, nbf);
//            //q_mtrx_.zero();
//
//            //if (gvec_.comm().rank() == 0) {
//            //    for (int xi2 = 0; xi2 < nbf; xi2++) {
//            //        for (int xi1 = 0; xi1 <= xi2; xi1++) {
//            //            /* packed orbital index */
//            //            int idx12         = utils::packed_index(xi1, xi2);
//            //            q_mtrx_(xi1, xi2) = q_mtrx_(xi2, xi1) = gvec_.omega() * q_pw_(idx12, 0);
//            //        }
//            //    }
//            //}
//            ///* broadcast from rank#0 */
//            //gvec_.comm().bcast(&q_mtrx_(0, 0), nbf * nbf, 0);
//
//            //if (atom_type_.parameters().cfg().control().print_checksum()) {
//            //    auto cs = q_pw_.checksum();
//            //    auto cs1 = q_mtrx_.checksum();
//            //    gvec_.comm().allreduce(&cs, 1);
//            //    if (gvec_.comm().rank() == 0) {
//            //        utils::print_checksum("q_pw", cs, std::cout);
//            //        utils::print_checksum("q_mtrx", cs1, std::cout);
//            //    }
//            //}
//
//        }
//
//    }
//
//  public:
//    Augmentation_operator_new(Unit_cell const& unit_cell__, sddk::Gvec const& gvec__,
//            Radial_integrals_aug<false> const& radial_integrals__)
//        : unit_cell_{unit_cell__}
//        , gvec_{gvec__}
//    {
//        init(radial_integrals__);
//        generate_pw_coeffs();
//        //generate_q_mtrx();
//    }
//
//
//};

/// Augmentation charge operator Q(r) of the ultrasoft pseudopotential formalism.
/** This class generates and stores the plane-wave coefficients of the augmentation charge operator for
    a given atom type. */
class Augmentation_operator
{
  private:
    Atom_type const& atom_type_;

    fft::Gvec const& gvec_;

    sddk::mdarray<double, 2> q_mtrx_;

    mutable sddk::mdarray<double, 2> q_pw_;

    mutable sddk::mdarray<double, 1> sym_weight_;

    sddk::mdarray<std::complex<double>, 1> zilm_;

    sddk::mdarray<double, 3> ri_values_;

    sddk::mdarray<int, 2> idx_;

    sddk::mdarray<int, 1> gvec_shell_;

    sddk::mdarray<int, 1> l_by_lm_;

    sddk::mdarray<double, 3> gaunt_coefs_;

  public:
    /// Constructor.
    /**\param [in] atom_type        Atom type instance.
     * \param [in] gvec             G-vector instance.
     * \param [in] radial_integrals Radial integrals of the Q(r) with spherical Bessel functions.
     */
    Augmentation_operator(Atom_type const& atom_type__, fft::Gvec const& gvec__,
        Radial_integrals_aug<false> const& radial_integrals__)
        : atom_type_(atom_type__)
        , gvec_(gvec__)
    {
        int lmax_beta = atom_type_.indexr().lmax();
        int lmax      = 2 * lmax_beta;
        int lmmax     = utils::lmmax(lmax);

        /* compute l of lm index */
        auto l_by_lm = utils::l_by_lm(lmax);
        l_by_lm_ = sddk::mdarray<int, 1>(lmmax);
        std::copy(l_by_lm.begin(), l_by_lm.end(), &l_by_lm_[0]);

        /* compute i^l array */
        zilm_ = sddk::mdarray<std::complex<double>, 1>(lmmax);
        for (int l = 0, lm = 0; l <= lmax; l++) {
            for (int m = -l; m <= l; m++, lm++) {
                zilm_[lm] = std::pow(std::complex<double>(0, 1), l);
            }
        }

        /* Gaunt coefficients of three real spherical harmonics */
        gaunt_coefs_ = Gaunt_coefficients<double>(lmax_beta, lmax, lmax_beta, SHT::gaunt_rrr).get_full_set_L3();

        /* number of beta-projectors */
        int nbf = atom_type_.mt_basis_size();
        /* number of beta-projector radial functions */
        int nbrf = atom_type__.mt_radial_basis_size();

        int idxmax = nbf * (nbf + 1) / 2;

        /* flatten the indices */
        idx_ = sddk::mdarray<int, 2>(3, idxmax);
        for (int xi2 = 0; xi2 < nbf; xi2++) {
            int lm2    = atom_type_.indexb(xi2).lm;
            int idxrf2 = atom_type_.indexb(xi2).idxrf;

            for (int xi1 = 0; xi1 <= xi2; xi1++) {
                int lm1    = atom_type_.indexb(xi1).lm;
                int idxrf1 = atom_type_.indexb(xi1).idxrf;

                /* packed orbital index */
                int idx12 = utils::packed_index(xi1, xi2);
                /* packed radial-function index */
                int idxrf12 = utils::packed_index(idxrf1, idxrf2);

                idx_(0, idx12) = lm1;
                idx_(1, idx12) = lm2;
                idx_(2, idx12) = idxrf12;
            }
        }

        /* local number of G-vectors for each rank */
        int gvec_count = gvec_.count();

        gvec_shell_ = sddk::mdarray<int, 1>(gvec_count);
        for (int igloc = 0; igloc < gvec_count; igloc++) {
            gvec_shell_(igloc) = gvec_.gvec_shell_idx_local(igloc);
        }

        ri_values_ = sddk::mdarray<double, 3>(nbrf * (nbrf + 1) / 2, lmax + 1, gvec_.num_gvec_shells_local());
        #pragma omp parallel for
        for (int j = 0; j < gvec_.num_gvec_shells_local(); j++) {
            auto ri = radial_integrals__.values(atom_type_.id(), gvec_.gvec_shell_len_local(j));
            for (int l = 0; l <= lmax; l++) {
                for (int i = 0; i < nbrf * (nbrf + 1) / 2; i++) {
                    ri_values_(i, l, j) = ri(i, l);
                }
            }
        }

        sym_weight_ = sddk::mdarray<double, 1>(idxmax);
        for (int xi2 = 0; xi2 < nbf; xi2++) {
            for (int xi1 = 0; xi1 <= xi2; xi1++) {
                /* packed orbital index */
                int idx12          = utils::packed_index(xi1, xi2);
                sym_weight_(idx12) = (xi1 == xi2) ? 1 : 2;
            }
        }

        if (atom_type_.parameters().processing_unit() == sddk::device_t::GPU) {
            auto& mpd = sddk::get_memory_pool(sddk::memory_t::device);
            l_by_lm_.allocate(mpd).copy_to(sddk::memory_t::device);
            zilm_.allocate(mpd).copy_to(sddk::memory_t::device);
            gaunt_coefs_.allocate(mpd).copy_to(sddk::memory_t::device);
            idx_.allocate(mpd).copy_to(sddk::memory_t::device);
            gvec_shell_.allocate(mpd).copy_to(sddk::memory_t::device);
            ri_values_.allocate(mpd).copy_to(sddk::memory_t::device);
            sym_weight_.allocate(mpd).copy_to(sddk::memory_t::device);
        }
    }

    void generate_pw_coeffs_chunk_gpu(int g_begin__, int ng__, double const* gvec_rlm__, int ld__,
            sddk::mdarray<double, 2>& qpw__) const
    {
#if defined(SIRIUS_GPU)
        double fourpi_omega = fourpi / gvec_.omega();

        /* maximum l of beta-projectors */
        int lmax_beta = atom_type_.indexr().lmax();
        int lmmax     = utils::lmmax(2 * lmax_beta);
        /* number of beta-projectors */
        int nbf = atom_type_.mt_basis_size();
        /* only half of Q_{xi,xi'}(G) matrix is stored */
        int nqlm = nbf * (nbf + 1) / 2;
        /* generate Q(G) */
        aug_op_pw_coeffs_gpu(ng__, gvec_shell_.at(sddk::memory_t::device, g_begin__), idx_.at(sddk::memory_t::device),
            nqlm, zilm_.at(sddk::memory_t::device), l_by_lm_.at(sddk::memory_t::device), lmmax,
            gaunt_coefs_.at(sddk::memory_t::device), static_cast<int>(gaunt_coefs_.size(0)),
            static_cast<int>(gaunt_coefs_.size(1)), gvec_rlm__, ld__,
            ri_values_.at(sddk::memory_t::device), static_cast<int>(ri_values_.size(0)),
            static_cast<int>(ri_values_.size(1)), qpw__.at(sddk::memory_t::device), static_cast<int>(qpw__.size(0)),
            fourpi_omega);
#endif
    }

    void generate_pw_coeffs();

    void prepare(stream_id sid) const // TODO: remove
    {
        if (atom_type_.parameters().processing_unit() == sddk::device_t::GPU && atom_type_.augment()) {
            q_pw_.allocate(sddk::get_memory_pool(sddk::memory_t::device));
            q_pw_.copy_to(sddk::memory_t::device, sid);
        }
    }

    void dismiss() const // TODO: remove
    {
        if (atom_type_.parameters().processing_unit() == sddk::device_t::GPU && atom_type_.augment()) {
            q_pw_.deallocate(sddk::memory_t::device);
        }
    }

    auto const& q_pw() const
    {
        return q_pw_;
    }

    double q_pw(int i__, int ig__) const
    {
        return q_pw_(i__, ig__);
    }

    /// Set Q-matrix.
    void q_mtrx(sddk::mdarray<double, 2> const& q_mtrx__) // TODO: remove
    {
        int nbf = atom_type_.mt_basis_size();
        for (int i = 0; i < nbf; i++) {
            for (int j = 0; j < nbf; j++) {
                q_mtrx_(j, i) = q_mtrx__(j, i);
            }
        }
    }

    /// Get values of the Q-matrix.
    inline double q_mtrx(int xi1__, int xi2__) const
    {
        return q_mtrx_(xi1__, xi2__);
    }

    inline auto const& sym_weight() const
    {
        return sym_weight_;
    }

    /// Weight of Q_{\xi,\xi'}.
    /** 2 if off-diagonal (xi != xi'), 1 if diagonal (xi=xi') */
    inline double sym_weight(int idx__) const
    {
        return sym_weight_(idx__);
    }

    Atom_type const& atom_type() const
    {
        return atom_type_;
    }
};

// TODO:
// can't cache it in the simulation context because each time the lattice is updated, PW coefficients must be
// recomputed; so, the only way to accelerate it is to move to GPUs..

/// Derivative of augmentation operator PW coefficients with respect to the Cartesian component of G-vector.
class Augmentation_operator_gvec_deriv
{
  private:
    fft::Gvec const& gvec_;

    sddk::mdarray<double, 2> q_pw_;

    sddk::mdarray<double, 1> sym_weight_;

    sddk::mdarray<double, 2> rlm_g_;

    sddk::mdarray<double, 3> rlm_dg_;

    sddk::mdarray<double, 3> ri_values_;

    sddk::mdarray<double, 3> ri_dg_values_;

    sddk::mdarray<int, 2> idx_;

    sddk::mdarray<double, 2> gvec_cart_;

    sddk::mdarray<int, 1> gvec_shell_;

    std::unique_ptr<Gaunt_coefficients<double>> gaunt_coefs_;

  public:
    Augmentation_operator_gvec_deriv(Simulation_parameters const& param__, int lmax__, fft::Gvec const& gvec__);

    void generate_pw_coeffs(Atom_type const& atom_type__, int nu__);

    void prepare(Atom_type const& atom_type__, Radial_integrals_aug<false> const& ri__,
        Radial_integrals_aug<true> const& ri_dq__);

    //void prepare(int stream_id__) const
    //{
    //    #ifdef SIRIUS_GPU
    //    if (atom_type_.parameters().processing_unit() == GPU && atom_type_.pp_desc().augment) {
    //        sym_weight_.allocate(memory_t::device);
    //        sym_weight_.async_copy_to_device(stream_id__);

    //        q_pw_.allocate(memory_t::device);
    //        q_pw_.async_copy_to_device(stream_id__);
    //    }
    //    #endif
    //}

    //void dismiss() const
    //{
    //    #ifdef SIRIUS_GPU
    //    if (atom_type_.parameters().processing_unit() == GPU && atom_type_.pp_desc().augment) {
    //        q_pw_.deallocate_on_device();
    //        sym_weight_.deallocate_on_device();
    //    }
    //    #endif
    //}

    auto const& q_pw() const
    {
        return q_pw_;
    }

    auto& q_pw()
    {
        return q_pw_;
    }

    double q_pw(int i__, int ig__) const
    {
        return q_pw_(i__, ig__);
    }

    /// Weight of Q_{\xi,\xi'}.
    /** 2 if off-diagonal (xi != xi'), 1 if diagonal (xi=xi') */
    inline double sym_weight(int idx__) const
    {
        return sym_weight_(idx__);
    }
};

} // namespace sirius

#endif // __AUGMENTATION_OPERATOR_H__
