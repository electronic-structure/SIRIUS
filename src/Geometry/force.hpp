// Copyright (c) 2013-2019 Anton Kozhevnikov, Ilia Sivkov, Thomas Schulthess
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

/** \file force.hpp
 *
 *  \brief Contains defintion of sirius::Force class.
 */

#ifndef __FORCE_HPP__
#define __FORCE_HPP__

#include "Hamiltonian/hamiltonian.hpp"

namespace sirius {
/* forward declaration */
class Simulation_context;
class Density;
class Potential;
class K_point;
class K_point_set;
class Hamiltonian_k;

/// Compute atomic forces.
class Force
{
  private:
    Simulation_context& ctx_;

    Density& density_;

    Potential& potential_;

    K_point_set& kset_;

    sddk::mdarray<double, 2> forces_vloc_;

    sddk::mdarray<double, 2> forces_us_;

    sddk::mdarray<double, 2> forces_nonloc_;

    sddk::mdarray<double, 2> forces_usnl_;

    sddk::mdarray<double, 2> forces_core_;

    sddk::mdarray<double, 2> forces_ewald_;

    sddk::mdarray<double, 2> forces_scf_corr_;

    sddk::mdarray<double, 2> forces_hubbard_;

    sddk::mdarray<double, 2> forces_hf_;

    sddk::mdarray<double, 2> forces_rho_;

    sddk::mdarray<double, 2> forces_ibs_;

    sddk::mdarray<double, 2> forces_total_;

    template <typename T>
    void add_k_point_contribution(K_point& kp__, sddk::mdarray<double, 2>& forces__) const;

    void symmetrize(sddk::mdarray<double, 2>& forces__) const;

    /** In the second-variational approach we need to compute the following expression for the k-dependent
     *  contribution to the forces:
     *  \f[
     *      {\bf F}_{\rm IBS}^{\alpha}=\sum_{\bf k}w_{\bf k}\sum_{l\sigma}n_{l{\bf k}}
     *      \sum_{ij}c_{\sigma i}^{l{\bf k}*}c_{\sigma j}^{l{\bf k}}
     *      {\bf F}_{ij}^{\alpha{\bf k}}
     *  \f]
     *  This function sums over band and spin indices to get the "density matrix":
     *  \f[
     *      q_{ij} = \sum_{l\sigma}n_{l{\bf k}} c_{\sigma i}^{l{\bf k}*}c_{\sigma j}^{l{\bf k}}
     *  \f]
     */
    void compute_dmat(K_point* kp__, sddk::dmatrix<double_complex>& dm__) const;

    /** Compute the forces for the simplex LDA+U method not the fully rotationally invariant one.
     *  It can not be used for LDA+U+SO either.
     *
     *  It is based on this reference : PRB 84, 161102(R) (2011)
     */
    void hubbard_force_add_k_contribution_colinear(K_point& kp__, Q_operator& q_op__, sddk::mdarray<double, 2>& forceh_);

    void add_ibs_force(K_point* kp__, Hamiltonian_k& Hk__, sddk::mdarray<double, 2>& ffac__, sddk::mdarray<double, 2>& forcek__) const;

  public:
    Force(Simulation_context& ctx__, Density& density__, Potential& potential__, K_point_set& kset__);

    sddk::mdarray<double, 2> const& calc_forces_vloc();

    inline sddk::mdarray<double, 2> const& forces_vloc() const
    {
        return forces_vloc_;
    }

    sddk::mdarray<double, 2> const& calc_forces_nonloc();

    inline sddk::mdarray<double, 2> const& forces_nonloc() const
    {
        return forces_nonloc_;
    }

    sddk::mdarray<double, 2> const& calc_forces_core();

    inline sddk::mdarray<double, 2> const& forces_core() const
    {
        return forces_core_;
    }

    sddk::mdarray<double, 2> const& calc_forces_scf_corr();

    inline sddk::mdarray<double, 2> const& forces_scf_corr() const
    {
        return forces_scf_corr_;
    }

    sddk::mdarray<double, 2> const& calc_forces_us();

    inline sddk::mdarray<double, 2> const& forces_us() const
    {
        return forces_us_;
    }

    sddk::mdarray<double, 2> const& calc_forces_ewald();

    sddk::mdarray<double, 2> const& forces_ewald() const
    {
        return forces_ewald_;
    }

    sddk::mdarray<double, 2> const& calc_forces_hubbard();

    inline sddk::mdarray<double, 2> const& forces_hubbard() const
    {
        return forces_hubbard_;
    }

    sddk::mdarray<double, 2> const& calc_forces_usnl();

    sddk::mdarray<double, 2> const& calc_forces_hf();

    inline sddk::mdarray<double, 2> const& forces_hf() const
    {
        return forces_hf_;
    }

    sddk::mdarray<double, 2> const& calc_forces_rho();

    inline sddk::mdarray<double, 2> const& forces_rho() const
    {
        return forces_rho_;
    }

    sddk::mdarray<double, 2> const& calc_forces_ibs();

    inline sddk::mdarray<double, 2> const& forces_ibs() const
    {
        return forces_ibs_;
    }

    sddk::mdarray<double, 2> const& calc_forces_total();

    inline sddk::mdarray<double, 2> const& forces_total() const
    {
        return forces_total_;
    }

    void print_info();
};

} // namespace sirius

#endif // __FORCE_HPP__
