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
#include "SDDK/gvec.hpp"

namespace sirius {

/// Augmentation charge operator Q(r) of the ultrasoft pseudopotential formalism.
/** This class generates and stores the plane-wave coefficients of the augmentation charge operator for
    a given atom type. */
class Augmentation_operator
{
  private:
    Atom_type const& atom_type_;

    sddk::Gvec const& gvec_;

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
    Augmentation_operator(Atom_type const& atom_type__, sddk::Gvec const& gvec__,
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
        int gvec_count  = gvec_.count();

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
        if (atom_type_.parameters().processing_unit() == sddk::device_t::GPU) {
            auto& mpd = sddk::get_memory_pool(sddk::memory_t::device);
            l_by_lm_.allocate(mpd).copy_to(sddk::memory_t::device);
            zilm_.allocate(mpd).copy_to(sddk::memory_t::device);
            gaunt_coefs_.allocate(mpd).copy_to(sddk::memory_t::device);
            idx_.allocate(mpd).copy_to(sddk::memory_t::device);
            gvec_shell_.allocate(mpd).copy_to(sddk::memory_t::device);
            ri_values_.allocate(mpd).copy_to(sddk::memory_t::device);
        }
    }

    void generate_pw_coeffs();

    void prepare(stream_id sid) const // TODO: q_pw is too big; must be handeled differently
    {
        if (atom_type_.parameters().processing_unit() == sddk::device_t::GPU && atom_type_.augment()) {
            sym_weight_.allocate(sddk::get_memory_pool(sddk::memory_t::device));
            q_pw_.allocate(sddk::get_memory_pool(sddk::memory_t::device));
            sym_weight_.copy_to(sddk::memory_t::device, sid);
            q_pw_.copy_to(sddk::memory_t::device, sid);
        }
    }

    void dismiss() const
    {
        if (atom_type_.parameters().processing_unit() == sddk::device_t::GPU && atom_type_.augment()) {
            q_pw_.deallocate(sddk::memory_t::device);
            sym_weight_.deallocate(sddk::memory_t::device);
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
    void q_mtrx(sddk::mdarray<double, 2> const& q_mtrx__)
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
    sddk::Gvec const& gvec_;

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
    Augmentation_operator_gvec_deriv(Simulation_parameters const& param__, int lmax__, sddk::Gvec const& gvec__);

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
