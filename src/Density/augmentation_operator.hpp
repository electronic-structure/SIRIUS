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

#include "radial_integrals.hpp"
#include "SDDK/gvec.hpp"

namespace sirius {

/// Augmentation charge operator Q(r) of the ultrasoft pseudopotential formalism.
/** This class generates and stores the plane-wave coefficients of the augmentation charge operator for
    a given atom type. */
class Augmentation_operator
{
  private:
    Atom_type const& atom_type_;

    Gvec const& gvec_;

    mdarray<double, 2> q_mtrx_;

    mutable mdarray<double, 2> q_pw_;

    mutable mdarray<double, 1> sym_weight_;

  public:
    Augmentation_operator(Atom_type const& atom_type__, Gvec const& gvec__)
        : atom_type_(atom_type__)
        , gvec_(gvec__)
    {
    }

    void generate_pw_coeffs(Radial_integrals_aug<false> const& radial_integrals__, sddk::mdarray<double, 2> const& tp__,
        memory_pool& mp__, memory_pool* mpd__);

    void prepare(stream_id sid, sddk::memory_pool* mp__) const
    {
        if (atom_type_.parameters().processing_unit() == device_t::GPU && atom_type_.augment()) {
            if (mp__) {
                sym_weight_.allocate(*mp__);
                q_pw_.allocate(*mp__);
            } else {
                sym_weight_.allocate(memory_t::device);
                q_pw_.allocate(memory_t::device);
            }
            sym_weight_.copy_to(memory_t::device, sid);
            q_pw_.copy_to(memory_t::device, sid);
        }
    }

    void dismiss() const
    {
        if (atom_type_.parameters().processing_unit() == device_t::GPU && atom_type_.augment()) {
            q_pw_.deallocate(memory_t::device);
            sym_weight_.deallocate(memory_t::device);
        }
    }

    mdarray<double, 2> const& q_pw() const
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

    inline mdarray<double, 1> const& sym_weight() const
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
// can't cache it in the simulation context becuase each time the lattice is updated, PW coefficients must be
// recomputed; so, the only way to accelerate it is to move to GPUs..

/// Derivative of augmentation operator PW coefficients with respect to the Cartesian component of G-vector.
class Augmentation_operator_gvec_deriv
{
  private:
    Gvec const& gvec_;

    Communicator const& comm_;

    mdarray<double, 2> q_pw_;

    mdarray<double, 1> sym_weight_;

    mdarray<double, 2> rlm_g_;

    mdarray<double, 3> rlm_dg_;

    std::unique_ptr<Gaunt_coefficients<double>> gaunt_coefs_;

  public:
    Augmentation_operator_gvec_deriv(int lmax__, Gvec const& gvec__, Communicator const& comm__)
        : gvec_(gvec__)
        , comm_(comm__)
    {
        PROFILE("sirius::Augmentation_operator_gvec_deriv");

        int lmax  = lmax__;
        int lmmax = utils::lmmax(2 * lmax);

        /* Gaunt coefficients of three real spherical harmonics */
        gaunt_coefs_ = std::unique_ptr<Gaunt_coefficients<double>>(new Gaunt_coefficients<double>(lmax, 2 * lmax, lmax, SHT::gaunt_rlm));

        /* split G-vectors between ranks */
        int gvec_count  = gvec__.count();

        rlm_g_  = mdarray<double, 2>(lmmax, gvec_count);
        rlm_dg_ = mdarray<double, 3>(lmmax, 3, gvec_count);

        /* array of real spherical harmonics and derivatives for each G-vector */
        #pragma omp parallel for schedule(static)
        for (int igloc = 0; igloc < gvec_count; igloc++) {
            auto gv = gvec__.gvec_cart<index_domain_t::local>(igloc);
            auto rtp = SHT::spherical_coordinates(gvec__.gvec_cart<index_domain_t::local>(igloc));

            double theta = rtp[1];
            double phi   = rtp[2];

            sf::spherical_harmonics(2 * lmax, theta, phi, &rlm_g_(0, igloc));

            sddk::mdarray<double, 2> tmp(&rlm_dg_(0, 0, igloc), lmmax, 3);
            sf::dRlm_dr(2 * lmax, gv, tmp, false);
        }
    }

    void generate_pw_coeffs(Atom_type                   const& atom_type__,
                            Radial_integrals_aug<false> const& ri__,
                            Radial_integrals_aug<true>  const& ri_dq__,
                            int                                nu__,
                            memory_pool&                       mp__)
    {
        PROFILE("sirius::Augmentation_operator_gvec_deriv::generate_pw_coeffs");

        /* maximum l of beta-projectors */
        int lmax_beta = atom_type__.indexr().lmax();
        int lmmax     = utils::lmmax(2 * lmax_beta);

        auto l_by_lm = utils::l_by_lm(2 * lmax_beta);

        std::vector<double_complex> zilm(lmmax);
        for (int l = 0, lm = 0; l <= 2 * lmax_beta; l++) {
            for (int m = -l; m <= l; m++, lm++) {
                zilm[lm] = std::pow(double_complex(0, 1), l);
            }
        }

        /* split G-vectors between ranks */
        int gvec_count  = gvec_.count();
        int gvec_offset = gvec_.offset();

        /* number of beta-projectors */
        int nbf = atom_type__.mt_basis_size();

        /* array of plane-wave coefficients */
        q_pw_ = mdarray<double, 2>(nbf * (nbf + 1) / 2, 2 * gvec_count, mp__, "q_pw_dg_");
        #pragma omp parallel for schedule(static)
        for (int igloc = 0; igloc < gvec_count; igloc++) {
            int    ig  = gvec_offset + igloc;
            double g   = gvec_.gvec_len(ig);
            auto   gvc = gvec_.gvec_cart<index_domain_t::local>(igloc);

            std::vector<double_complex> v(lmmax);

            auto ri    = ri__.values(atom_type__.id(), g);
            auto ri_dg = ri_dq__.values(atom_type__.id(), g);

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

                    for (int lm3 = 0; lm3 < lmmax; lm3++) {
                        v[lm3] = std::conj(zilm[lm3]) * (rlm_dg_(lm3, nu__, igloc) * ri(idxrf12, l_by_lm[lm3]) +
                                                         rlm_g_(lm3, igloc) * ri_dg(idxrf12, l_by_lm[lm3]) * gvc[nu__]);
                    }

                    double_complex z = fourpi * gaunt_coefs_->sum_L3_gaunt(lm2, lm1, &v[0]);

                    q_pw_(idx12, 2 * igloc)     = z.real();
                    q_pw_(idx12, 2 * igloc + 1) = z.imag();
                }
            }
        }

        memory_t mem{memory_t::host};
        if (atom_type__.parameters().processing_unit() == device_t::GPU) {
            mem = memory_t::host_pinned;
        }
        sym_weight_ = mdarray<double, 1>(nbf * (nbf + 1) / 2, mem, "sym_weight_");
        for (int xi2 = 0; xi2 < nbf; xi2++) {
            for (int xi1 = 0; xi1 <= xi2; xi1++) {
                /* packed orbital index */
                int idx12          = xi2 * (xi2 + 1) / 2 + xi1;
                sym_weight_(idx12) = (xi1 == xi2) ? 1 : 2;
            }
        }
    }

    //void prepare(int stream_id__) const
    //{
    //    #ifdef __GPU
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
    //    #ifdef __GPU
    //    if (atom_type_.parameters().processing_unit() == GPU && atom_type_.pp_desc().augment) {
    //        q_pw_.deallocate_on_device();
    //        sym_weight_.deallocate_on_device();
    //    }
    //    #endif
    //}

    mdarray<double, 2> const& q_pw() const
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
