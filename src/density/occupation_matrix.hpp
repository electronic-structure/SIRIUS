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
    Occupation_matrix(Simulation_context& ctx__);

    template <typename T>
    void
    add_k_point_contribution(K_point<T>& kp__);

    /** The initial occupancy is calculated following Hund rules. We first
     *  fill the d (f) states according to the hund's rules and with majority
     *  spin first and the remaining electrons distributed among the minority states. */
    void
    init();

    void
    reduce()
    {
        if (!ctx_.hubbard_correction()) {
            return;
        }

        /* global reduction over k points */
        for (int at_lvl = 0; at_lvl < (int)this->local_.size(); at_lvl++) {
            const int ia     = atomic_orbitals_[at_lvl].first;
            auto const& atom = ctx_.unit_cell().atom(ia);
            if (atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).use_for_calculation()) {
                ctx_.comm_k().allreduce(this->local(at_lvl).at(memory_t::host),
                                        static_cast<int>(this->local(at_lvl).size()));
            }
        }

        /* reduce occ_mtrx_T_ (not nonlocal - it is computed during symmetrization from occ_mtrx_T_) */
        for (auto& e : this->occ_mtrx_T_) {
            ctx_.comm_k().allreduce(e.second.at(memory_t::host), static_cast<int>(e.second.size()));
        }
    }

    void
    update_nonlocal()
    {
        if (ctx_.num_mag_dims() == 3) {
            RTE_THROW("only collinear case is supported");
        }
        for (int i = 0; i < static_cast<int>(ctx_.cfg().hubbard().nonlocal().size()); i++) {
            auto nl = ctx_.cfg().hubbard().nonlocal(i);
            int ia  = nl.atom_pair()[0];
            int ja  = nl.atom_pair()[1];
            int il  = nl.l()[0];
            int jl  = nl.l()[1];
            int n1  = nl.n()[0];
            int n2  = nl.n()[1];
            int ib  = 2 * il + 1;
            int jb  = 2 * jl + 1;
            auto T  = nl.T();
            this->nonlocal(i).zero();

            /* NOTE : the atom order is important here. */
            int at1_lvl          = this->find_orbital_index(ia, n1, il);
            int at2_lvl          = this->find_orbital_index(ja, n2, jl);
            auto const& occ_mtrx = occ_mtrx_T_.at(T);

            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                for (int m1 = 0; m1 < ib; m1++) {
                    for (int m2 = 0; m2 < jb; m2++) {
                        this->nonlocal(i)(m1, m2, ispn) =
                                occ_mtrx(this->offset(at1_lvl) + m1, this->offset(at2_lvl) + m2, ispn);
                    }
                }
            }
        }
    }

    void
    zero()
    {
        Hubbard_matrix::zero();
        for (auto& e : occ_mtrx_T_) {
            e.second.zero();
        }
    }

    void
    calculate_constraints_and_error();
    void
    print_occupancies(int verbosity__) const;

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
