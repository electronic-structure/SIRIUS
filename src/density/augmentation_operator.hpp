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

    Gvec const& gvec_;

    sddk::mdarray<double, 2> q_mtrx_;

    mutable sddk::mdarray<double, 2> q_pw_;

    mutable sddk::mdarray<double, 1> sym_weight_;

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

    sddk::mdarray<double, 2> q_pw_;

    sddk::mdarray<double, 1> sym_weight_;

    sddk::mdarray<double, 2> rlm_g_;

    sddk::mdarray<double, 3> rlm_dg_;

    std::unique_ptr<Gaunt_coefficients<double>> gaunt_coefs_;

  public:
    Augmentation_operator_gvec_deriv(int lmax__, Gvec const& gvec__, sddk::mdarray<double, 2> const& tp__);

    void generate_pw_coeffs(Atom_type const& atom_type__, Radial_integrals_aug<false> const& ri__,
        Radial_integrals_aug<true> const& ri_dq__, int nu__, memory_pool& mp__, memory_pool* mpd__);

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
