/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file occupation_matrix.hpp
 *
 *  \brief Occupation matrix of the LDA+U method.
 */

#ifndef __OCCUPATION_MATRIX_HPP__
#define __OCCUPATION_MATRIX_HPP__

#include "k_point/k_point.hpp"
#include "hubbard/hubbard_matrix.hpp"

namespace sirius {

class Occupation_matrix : public Hubbard_matrix
{
  private:
    /// K-point contribution to density matrices weighted with e^{ikT} phase factors.
    std::map<r3::vector<int>, mdarray<std::complex<double>, 3>> occ_mtrx_T_;

  public:
    /// Constructor.
    Occupation_matrix(Simulation_context& ctx__);

    /// Add contribution from k-point.
    template <typename T>
    void
    add_k_point_contribution(K_point<T>& kp__);

    /** The initial occupancy is calculated following Hund rules. We first
     *  fill the d (f) states according to the hund's rules and with majority
     *  spin first and the remaining electrons distributed among the minority states. */
    void
    init();

    /// Sum over k-points.
    void
    reduce();

    /// Copy non-local block corresponding to a pair of atoms from occ_mtrx_T_ to this->nonlocal.
    /** If symmetrization is not performed, the non-local blocks of occupation matrix must be copied from
     *  occ_mtrx_T to this->nonlocal. */
    void
    update_nonlocal();

    void
    calculate_constraints_and_error();

    void
    print_occupancies(int verbosity__) const;

    /// Zero occupation matrix.
    void
    zero()
    {
        Hubbard_matrix::zero();
        for (auto& e : occ_mtrx_T_) {
            e.second.zero();
        }
    }

    inline auto const&
    occ_mtrx_T(r3::vector<int> T__) const
    {
        return occ_mtrx_T_.at(T__);
    }

    inline auto const&
    occ_mtrx_T() const
    {
        return occ_mtrx_T_;
    }

    friend void
    copy(Occupation_matrix const& src__, Occupation_matrix& dest__);
};

inline void
copy(Occupation_matrix const& src__, Occupation_matrix& dest__)
{
    for (int at_lvl = 0; at_lvl < static_cast<int>(src__.atomic_orbitals().size()); at_lvl++) {
        copy(src__.local(at_lvl), dest__.local(at_lvl));
    }

    for (int i = 0; i < static_cast<int>(src__.ctx().cfg().hubbard().nonlocal().size()); i++) {
        copy(src__.nonlocal(i), dest__.nonlocal(i));
    }

    for (auto& e : src__.occ_mtrx_T()) {
        copy(e.second, dest__.occ_mtrx_T_.at(e.first));
    }

    for (int i = 0; i < static_cast<int>(src__.local_constraints().size()); i++) {
        copy(src__.local_constraints(i), dest__.local_constraints(i));
    }

    for (int i = 0; i < static_cast<int>(src__.multipliers_constraints().size()); i++) {
        copy(src__.multipliers_constraints(i), dest__.multipliers_constraints(i));
    }

    for (int i = 0; i < static_cast<int>(src__.apply_constraints().size()); i++) {
        dest__.apply_constraints()[i] = src__.apply_constraints(i);
    }

    dest__.constraint_error()                = src__.constraint_error();
    dest__.constraint_number_of_iterations() = src__.constraint_number_of_iterations();
}

} // namespace sirius
#endif
