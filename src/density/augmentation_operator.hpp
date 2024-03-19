/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file augmentation_operator.hpp
 *
 *  \brief Contains implementation of sirius::Augmentation_operator class.
 */

#ifndef __AUGMENTATION_OPERATOR_HPP__
#define __AUGMENTATION_OPERATOR_HPP__

#include "radial/radial_integrals.hpp"
#include "core/fft/gvec.hpp"

#if defined(SIRIUS_GPU)
extern "C" {

void
aug_op_pw_coeffs_gpu(int ngvec__, int const* gvec_shell__, int const* idx__, int idxmax__,
                     std::complex<double> const* zilm__, int const* l_by_lm__, int lmmax__, double const* gc__,
                     int ld0__, int ld1__, double const* gvec_rlm__, int ld2__, double const* ri_values__, int ld3__,
                     int ld4__, double* q_pw__, int ld5__, double fourpi_omega__);

void
aug_op_pw_coeffs_deriv_gpu(int ngvec__, int const* gvec_shell__, double const* gvec_cart__, int const* idx__,
                           int idxmax__, double const* gc__, int ld0__, int ld1__, double const* rlm__,
                           double const* rlm_dg__, int ld2__, double const* ri_values__, double const* ri_dg_values__,
                           int ld3__, int ld4__, double* q_pw__, int ld5__, double fourpi__, int nu__, int lmax_q__);

void
spherical_harmonics_rlm_gpu(int lmax__, int ntp__, double const* theta__, double const* phi__, double* rlm__, int ld__);
}
#endif

namespace sirius {

template <typename F>
inline void
iterate_aug_atom_types(Unit_cell const& uc__, F&& f__)
{
    for (int iat = 0; iat < uc__.num_atom_types(); iat++) {
        auto& atom_type = uc__.atom_type(iat);

        if (!atom_type.augment() || atom_type.num_atoms() == 0) {
            continue;
        }
        f__(atom_type);
    }
}

inline auto
max_l_aug(Unit_cell const& uc__)
{
    int l{0};

    iterate_aug_atom_types(uc__, [&l](Atom_type const& type__) { l = std::max(l, type__.indexr().lmax()); });

    return l;
}

inline auto
max_na_aug(Unit_cell const& uc__)
{
    int na{0};

    iterate_aug_atom_types(uc__, [&na](Atom_type const& type__) { na = std::max(na, type__.num_atoms()); });

    return na;
}

inline auto
max_nb_aug(Unit_cell const& uc__)
{
    int nb{0};

    iterate_aug_atom_types(uc__, [&nb](Atom_type const& type__) { nb = std::max(nb, type__.mt_basis_size()); });

    return nb;
}

/// Augmentation charge operator Q(r) of the ultrasoft pseudopotential formalism.
/** This class generates and stores the plane-wave coefficients of the augmentation charge operator for
    a given atom type. */
class Augmentation_operator
{
  private:
    Atom_type const& atom_type_;

    fft::Gvec const& gvec_;

    mdarray<double, 2> q_mtrx_;

    mdarray<double, 2> q_pw_;

    mdarray<double, 1> sym_weight_;

    mdarray<std::complex<double>, 1> zilm_;

    mdarray<double, 3> ri_values_;
    mdarray<double, 3> ri_dq_values_;

    mdarray<int, 2> idx_;

    mdarray<int, 1> l_by_lm_;

  public:
    /// Constructor.
    /**\param [in] atom_type        Atom type instance.
     * \param [in] gvec             G-vector instance.
     * \param [in] radial_integrals Radial integrals of the Q(r) with spherical Bessel functions.
     */
    Augmentation_operator(Atom_type const& atom_type__, fft::Gvec const& gvec__,
                          Radial_integrals_aug<false> const& ri__, Radial_integrals_aug<true> const& ri_dq__)
        : atom_type_(atom_type__)
        , gvec_(gvec__)
    {
        int lmax_beta = atom_type_.indexr().lmax();
        int lmax      = 2 * lmax_beta;
        int lmmax     = sf::lmmax(lmax);

        /* compute l of lm index */
        auto l_by_lm = sf::l_by_lm(lmax);
        l_by_lm_     = mdarray<int, 1>({lmmax});
        std::copy(l_by_lm.begin(), l_by_lm.end(), &l_by_lm_[0]);

        /* compute i^l array */
        zilm_ = mdarray<std::complex<double>, 1>({lmmax});
        for (int l = 0, lm = 0; l <= lmax; l++) {
            for (int m = -l; m <= l; m++, lm++) {
                zilm_[lm] = std::pow(std::complex<double>(0, 1), l);
            }
        }

        /* number of beta-projectors */
        int nbf = atom_type_.mt_basis_size();
        /* number of beta-projector radial functions */
        int nbrf = atom_type__.mt_radial_basis_size();
        /* only half of Q_{xi,xi'}(G) matrix is stored */
        int nqlm = nbf * (nbf + 1) / 2;

        /* flatten the indices */
        idx_ = mdarray<int, 2>({3, nqlm});
        for (int xi2 = 0; xi2 < nbf; xi2++) {
            int lm2    = atom_type_.indexb(xi2).lm;
            int idxrf2 = atom_type_.indexb(xi2).idxrf;

            for (int xi1 = 0; xi1 <= xi2; xi1++) {
                int lm1    = atom_type_.indexb(xi1).lm;
                int idxrf1 = atom_type_.indexb(xi1).idxrf;

                /* packed orbital index */
                int idx12 = packed_index(xi1, xi2);
                /* packed radial-function index */
                int idxrf12 = packed_index(idxrf1, idxrf2);

                idx_(0, idx12) = lm1;
                idx_(1, idx12) = lm2;
                idx_(2, idx12) = idxrf12;
            }
        }

        ri_values_    = mdarray<double, 3>({nbrf * (nbrf + 1) / 2, lmax + 1, gvec_.num_gvec_shells_local()});
        ri_dq_values_ = mdarray<double, 3>({nbrf * (nbrf + 1) / 2, lmax + 1, gvec_.num_gvec_shells_local()});
        #pragma omp parallel for
        for (int j = 0; j < gvec_.num_gvec_shells_local(); j++) {
            auto ri    = ri__.values(atom_type_.id(), gvec_.gvec_shell_len_local(j));
            auto ri_dq = ri_dq__.values(atom_type__.id(), gvec_.gvec_shell_len_local(j));
            for (int l = 0; l <= lmax; l++) {
                for (int i = 0; i < nbrf * (nbrf + 1) / 2; i++) {
                    ri_values_(i, l, j)    = ri(i, l);
                    ri_dq_values_(i, l, j) = ri_dq(i, l);
                }
            }
        }

        sym_weight_ = mdarray<double, 1>({nqlm});
        for (int xi2 = 0; xi2 < nbf; xi2++) {
            for (int xi1 = 0; xi1 <= xi2; xi1++) {
                /* packed orbital index */
                int idx12          = packed_index(xi1, xi2);
                sym_weight_(idx12) = (xi1 == xi2) ? 1 : 2;
            }
        }

        if (atom_type_.parameters().processing_unit() == device_t::GPU) {
            auto& mpd = get_memory_pool(memory_t::device);
            sym_weight_.allocate(mpd).copy_to(memory_t::device);
        }

        /* allocate array of plane-wave coefficients */
        auto mt = (atom_type_.parameters().processing_unit() == device_t::CPU) ? memory_t::host : memory_t::host_pinned;
        q_pw_   = mdarray<double, 2>({nqlm, 2 * gvec_.count()}, get_memory_pool(mt), mdarray_label("q_pw_"));
    }

    // TODO: not used at the moment, evaluate the possibility to remove in the future
    //    /// Generate chunk of plane-wave coefficients on the GPU.
    //    void generate_pw_coeffs_chunk_gpu(int g_begin__, int ng__, double const* gvec_rlm__, int ld__,
    //            mdarray<double, 2>& qpw__) const
    //    {
    // #if defined(SIRIUS_GPU)
    //        double fourpi_omega = fourpi / gvec_.omega();
    //
    //        /* maximum l of beta-projectors */
    //        int lmax_beta = atom_type_.indexr().lmax();
    //        int lmmax     = sf::lmmax(2 * lmax_beta);
    //        /* number of beta-projectors */
    //        int nbf = atom_type_.mt_basis_size();
    //        /* only half of Q_{xi,xi'}(G) matrix is stored */
    //        int nqlm = nbf * (nbf + 1) / 2;
    //        /* generate Q(G) */
    //        aug_op_pw_coeffs_gpu(ng__, gvec_shell_.at(memory_t::device, g_begin__), idx_.at(memory_t::device),
    //            nqlm, zilm_.at(memory_t::device), l_by_lm_.at(memory_t::device), lmmax,
    //            gaunt_coefs_.at(memory_t::device), static_cast<int>(gaunt_coefs_.size(0)),
    //            static_cast<int>(gaunt_coefs_.size(1)), gvec_rlm__, ld__,
    //            ri_values_.at(memory_t::device), static_cast<int>(ri_values_.size(0)),
    //            static_cast<int>(ri_values_.size(1)), qpw__.at(memory_t::device), static_cast<int>(qpw__.size(0)),
    //            fourpi_omega);
    // #endif
    //    }

    /// Generate Q_{xi,xi'}(G) plane wave coefficients.
    void
    generate_pw_coeffs();

    /// Generate G-vector derivative Q_{xi,xi'}(G)/dG of the plane-wave coefficients */
    void
    generate_pw_coeffs_gvec_deriv(int nu__);

    auto const&
    q_pw() const
    {
        return q_pw_;
    }

    double
    q_pw(int i__, int ig__) const
    {
        return q_pw_(i__, ig__);
    }

    /// Get values of the Q-matrix.
    inline double
    q_mtrx(int xi1__, int xi2__) const
    {
        return q_mtrx_(xi1__, xi2__);
    }

    inline auto const&
    sym_weight() const
    {
        return sym_weight_;
    }

    /// Weight of Q_{\xi,\xi'}.
    /** 2 if off-diagonal (xi != xi'), 1 if diagonal (xi=xi') */
    inline double
    sym_weight(int idx__) const
    {
        return sym_weight_(idx__);
    }

    Atom_type const&
    atom_type() const
    {
        return atom_type_;
    }
};

} // namespace sirius

#endif // __AUGMENTATION_OPERATOR_H__
