/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file hubbard_matrix.hpp
 *
 *  \brief Base class for Hubbard occupancy and potential matrices.
 */

#ifndef __HUBBARD_MATRIX_HPP__
#define __HUBBARD_MATRIX_HPP__

#include "context/simulation_context.hpp"

namespace sirius {

/// Describes Hubbard orbital occupancy or potential correction matrices.
class Hubbard_matrix
{
  protected:
    Simulation_context const& ctx_;
    int num_steps_{0};
    double constraint_error_{1.0};
    int num_atomic_levels_{0};
    /// Table indicating if we should apply constraints on the hubbard occupation
    /// to given atomic orbital group
    std::vector<bool> active_constraints_;
    /// Local part of Hubbard matrix
    std::vector<mdarray<std::complex<double>, 3>> local_;
    /// Non-local part of Hubbard matrix.
    std::vector<mdarray<std::complex<double>, 3>> nonlocal_;
    /// occupancy matrix for each atomic orbital (n,l) contributing to the hubbard correction
    std::vector<mdarray<std::complex<double>, 3>> local_constraints_;
    /// "Lagrange" multipliers
    std::vector<mdarray<std::complex<double>, 3>> multipliers_constraints_;
    std::vector<std::pair<int, int>> atomic_orbitals_;
    std::vector<int> offset_;

  public:
    Hubbard_matrix(Simulation_context& ctx__);

    /// Retrieve or set elements of the Hubbard matrix.
    /** This functions helps retrieving or setting up the hubbard occupancy
     *  tensors from an external tensor. Retrieving it is done by specifying
     *  "get" in the first argument of the method while setting it is done
     *  with the parameter set up to "set". The second parameter is the
     *  output pointer and the last parameter is the leading dimension of the
     *  tensor.
     *
     *  The returned result has the same layout than SIRIUS layout, * i.e.,
     *  the harmonic orbitals are stored from m_z = -l..l. The occupancy
     *  matrix can also be accessed through the method occupation_matrix()
     *
     * \param [in]    what String to set to "set" for initializing sirius ccupancy tensor and "get" for retrieving it.
     * \param [inout] occ  Pointer to external occupancy tensor.
     * \param [in]    ld   Leading dimension of the outside tensor.
     * \return return the occupancy matrix if the first parameter is set to "get". */
    void
    access(std::string const& what__, std::complex<double>* ptr__, int ld__);

    void
    print_local(int idx__, std::ostream& out__) const;

    void
    print_nonlocal(int idx__, std::ostream& out__) const;

    void
    print(std::ostream& out__) const;

    void
    zero();

    /// Return local occupation matrix for a given composite atomic level (atom + local n,l).
    auto&
    local(int idx__)
    {
        return local_[idx__];
    }

    /// Return const reference to local occupation matrix block.
    auto const&
    local(int idx__) const
    {
        return local_[idx__];
    }

    auto&
    nonlocal(int idx__)
    {
        return nonlocal_[idx__];
    }

    auto const&
    nonlocal(int idx__) const
    {
        return nonlocal_[idx__];
    }

    const auto&
    atomic_orbital(const int idx__) const
    {
        return atomic_orbitals_[idx__];
    }

    int
    num_steps() const
    {
        return num_steps_;
    }

    int
    num_steps(const int num_steps__)
    {
        num_steps_ = num_steps__;
        return num_steps_;
    }

    double
    constraint_error() const
    {
        return constraint_error_;
    }

    int
    offset(const int idx__) const
    {
        return offset_[idx__];
    }

    auto const&
    local_constraint(int idx__) const
    {
        return local_constraints_[idx__];
    }

    auto const&
    active_constraints() const
    {
        return active_constraints_;
    }

    bool
    active_constraint(int idx__) const
    {
        return active_constraints_[idx__];
    }

    auto&
    multipliers_constraint(int idx__)
    {
        return multipliers_constraints_[idx__];
    }

    auto const&
    multipliers_constraint(int idx__) const
    {
        return multipliers_constraints_[idx__];
    }

    bool
    apply_constraints() const
    {
        return (this->constraint_error_ > ctx_.cfg().hubbard().constraint().error()) &&
               (this->num_steps_ < ctx_.cfg().hubbard().constraint().maxiter()) &&
               ctx_.hubbard_constrained_calculation();
    }

    auto const&
    ctx() const
    {
        return ctx_;
    }

    auto
    find_orbital_index(const int ia__, const int n__, const int l__) const
    {
        for (int at_lvl = 0; at_lvl < static_cast<int>(atomic_orbitals_.size()); at_lvl++) {
            int lo_ind = atomic_orbitals_[at_lvl].second;

            if ((atomic_orbitals_[at_lvl].first == ia__) &&
                (ctx_.unit_cell().atom(ia__).type().lo_descriptor_hub(lo_ind).n() == n__) &&
                (ctx_.unit_cell().atom(ia__).type().lo_descriptor_hub(lo_ind).l() == l__)) {
                return at_lvl;
            }
        }

        std::stringstream s;
        s << "Atomic orbital is not in the list" << std::endl
          << "  atom: " << ia__ << ", n: " << n__ << ", l: " << l__ << std::endl
          << "  list of atomic orbitals for a given atom:" << std::endl;
        for (int at_lvl = 0; at_lvl < static_cast<int>(atomic_orbitals_.size()); at_lvl++) {
            int lo_ind = atomic_orbitals_[at_lvl].second;
            if (atomic_orbitals_[at_lvl].first == ia__) {
                s << "  at_lvl: " << at_lvl
                  << ", n: " << ctx_.unit_cell().atom(ia__).type().lo_descriptor_hub(lo_ind).n()
                  << ", l: " << ctx_.unit_cell().atom(ia__).type().lo_descriptor_hub(lo_ind).l() << std::endl;
            }
        }
        RTE_THROW(s);

        return -1;
    }

    inline auto
    local_checksum() const
    {
        std::complex<double> sum(0, 0);
        for (auto& e : local_) {
            sum += e.checksum();
        }
        return sum;
    }

    inline auto
    nonlocal_checksum() const
    {
        std::complex<double> sum(0, 0);
        for (auto& e : nonlocal_) {
            sum += e.checksum();
        }
        return sum;
    }

    /// Number of atomic levels to which U correction is applied.
    /** Defines the size of local, active_constraints, local_constraints, and
     *  multipliers_constraints arrays. */
    inline auto
    num_atomic_levels() const
    {
        return num_atomic_levels_;
    }

    inline auto
    num_nonlocal() const
    {
        return static_cast<int>(nonlocal_.size());
    }
};

inline void
copy(Hubbard_matrix const& src__, Hubbard_matrix& dest__)
{
    for (int at_lvl = 0; at_lvl < src__.num_atomic_levels(); at_lvl++) {
        copy(src__.local(at_lvl), dest__.local(at_lvl));
    }

    for (int i = 0; i < src__.num_nonlocal(); i++) {
        copy(src__.nonlocal(i), dest__.nonlocal(i));
    }

    if (src__.ctx().hubbard_constrained_calculation()) {
        for (int at_lvl = 0; at_lvl < src__.num_atomic_levels(); at_lvl++) {
            copy(src__.multipliers_constraint(at_lvl), dest__.multipliers_constraint(at_lvl));
        }
    }
}

inline void
axpy(const double alpha__, Hubbard_matrix const& src__, Hubbard_matrix& dest__)
{
    for (int at_lvl = 0; at_lvl < src__.num_atomic_levels(); at_lvl++) {
        for (size_t i = 0; i < src__.local(at_lvl).size(); i++) {
            dest__.local(at_lvl)[i] = alpha__ * src__.local(at_lvl)[i] + dest__.local(at_lvl)[i];
        }
    }
    for (int j = 0; j < src__.num_nonlocal(); j++) {
        for (size_t i = 0; i < src__.nonlocal(j).size(); i++) {
            dest__.nonlocal(j)[i] = alpha__ * src__.nonlocal(j)[i] + dest__.nonlocal(j)[i];
        }
    }

    if (src__.ctx().hubbard_constrained_calculation()) {
        for (int at_lvl = 0; at_lvl < src__.num_atomic_levels(); at_lvl++) {
            for (size_t i = 0; i < src__.multipliers_constraint(at_lvl).size(); i++) {
                dest__.multipliers_constraint(at_lvl)[i] =
                        alpha__ * src__.multipliers_constraint(at_lvl)[i] + dest__.multipliers_constraint(at_lvl)[i];
            }
        }
    }
}

inline void
rotate(double c__, double s__, Hubbard_matrix& src__, Hubbard_matrix& dest__)
{
    for (int at_lvl = 0; at_lvl < src__.num_atomic_levels(); at_lvl++) {
        for (size_t i = 0; i < src__.local(at_lvl).size(); i++) {
            auto xi                 = src__.local(at_lvl)[i];
            auto yi                 = dest__.local(at_lvl)[i];
            src__.local(at_lvl)[i]  = xi * c__ + yi * s__;
            dest__.local(at_lvl)[i] = yi * c__ - xi * s__;
        }
    }

    for (int j = 0; j < src__.num_nonlocal(); j++) {
        for (size_t i = 0; i < src__.nonlocal(j).size(); i++) {
            auto xi                    = src__.nonlocal(j)[i];
            auto yi                    = dest__.nonlocal(j)[i];
            src__.nonlocal(j)[i]  = xi * c__ + yi * s__;
            dest__.nonlocal(j)[i] = yi * c__ - xi * s__;
        }
    }

    if (src__.ctx().hubbard_constrained_calculation()) {
        for (int at_lvl = 0; at_lvl < src__.num_atomic_levels(); at_lvl++) {
            for (size_t i = 0; i < src__.multipliers_constraint(at_lvl).size(); i++) {
                auto xi                                   = src__.multipliers_constraint(at_lvl)[i];
                auto yi                                   = dest__.multipliers_constraint(at_lvl)[i];
                src__.multipliers_constraint(at_lvl)[i]  = xi * c__ + yi * s__;
                dest__.multipliers_constraint(at_lvl)[i] = yi * c__ - xi * s__;
            }
        }
    }
}

inline void
scale(double alpha__, Hubbard_matrix& src__)
{
    for (int at_lvl = 0; at_lvl < src__.num_atomic_levels(); at_lvl++) {
        for (size_t i = 0; i < src__.local(at_lvl).size(); i++) {
            src__.local(at_lvl)[i] *= alpha__;
        }
    }

    for (int j = 0; j < src__.num_nonlocal(); j++) {
        for (size_t i = 0; i < src__.nonlocal(j).size(); i++) {
            src__.nonlocal(j)[i] *= alpha__;
        }
    }

    if (src__.ctx().hubbard_constrained_calculation()) {
        for (int at_lvl = 0; at_lvl < src__.num_atomic_levels(); at_lvl++) {
            for (size_t i = 0; i < src__.multipliers_constraint(at_lvl).size(); i++) {
                src__.multipliers_constraint(at_lvl)[i] *= alpha__;
            }
        }
    }
}

} // namespace sirius

#endif
