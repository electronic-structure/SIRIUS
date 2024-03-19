/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file atom_symmetry_class.hpp
 *
 *  \brief Contains declaration and partial implementation of sirius::Atom_symmetry_class class.
 */

#ifndef __ATOM_SYMMETRY_CLASS_HPP__
#define __ATOM_SYMMETRY_CLASS_HPP__

#include "atom_type.hpp"
#include "core/la/eigensolver.hpp"
#include "core/mpi/pstdout.hpp"

namespace sirius {

/// Data and methods specific to the symmetry class of the atom.
/** Atoms transforming into each other under symmetry opeartions belong to the same symmetry class. They have the
 *  same spherical part of the on-site potential and, as a consequence, the same radial functions.
 */
class Atom_symmetry_class
{
  private:
    /// Symmetry class id in the range [0, N_class).
    int id_;

    /// List of atoms of this class.
    std::vector<int> atom_id_;

    /// Pointer to atom type.
    Atom_type const& atom_type_;

    /// Spherical part of the effective potential.
    /** Used by the LAPW radial solver. */
    std::vector<double> spherical_potential_;

    /// List of radial functions for the LAPW basis.
    /** This array stores all the radial functions (AW and LO) and their derivatives. Radial derivatives of functions
     *  are multiplied by \f$ x \f$.\n
     *  1-st dimension: index of radial point \n
     *  2-nd dimension: index of radial function \n
     *  3-nd dimension: 0 - function itself, 1 - radial derivative r*(du/dr) */
    mdarray<double, 3> radial_functions_;

    /// Surface derivatives of AW radial functions.
    mdarray<double, 2> surface_derivatives_;

    /// Spherical part of radial integral.
    mdarray<double, 2> h_spherical_integrals_;

    /// Overlap integrals.
    mdarray<double, 3> o_radial_integrals_;

    /// Overlap integrals for IORA relativistic treatment.
    mdarray<double, 2> o1_radial_integrals_;

    /// Spin-orbit interaction integrals.
    mdarray<double, 3> so_radial_integrals_;

    /// Core charge density.
    /** All-electron core charge density of the LAPW method. It is recomputed on every SCF iteration due to
        the change of effective potential. */
    std::vector<double> ae_core_charge_density_;

    /// Core eigen-value sum.
    double core_eval_sum_{0};

    /// Core leakage.
    double core_leakage_{0};

    /// list of radial descriptor sets used to construct augmented waves
    mutable std::vector<radial_solution_descriptor_set> aw_descriptors_;

    /// list of radial descriptor sets used to construct local orbitals
    mutable std::vector<local_orbital_descriptor> lo_descriptors_;

    /// Generate radial functions for augmented waves
    int
    generate_aw_radial_functions(relativity_t rel__, mdarray<double, 3>& rf__, mdarray<double, 2>& sd__) const;

    /// Generate local orbital raidal functions
    int
    generate_lo_radial_functions(relativity_t rel__, mdarray<double, 3>& rf__) const;

    /// Orthogonalize the radial functions.
    void
    orthogonalize_radial_functions();

  public:
    /// Constructor
    Atom_symmetry_class(int id_, Atom_type const& atom_type_);

    /// Set the spherical component of the potential
    /** Atoms belonging to the same symmetry class have the same spherical potential. */
    void
    set_spherical_potential(std::vector<double> const& vs__);

    /// Generate APW and LO radial functions.
    void
    generate_radial_functions(relativity_t rel__);

    void
    sync_radial_functions(mpi::Communicator const& comm__, int const rank__);

    void
    sync_radial_integrals(mpi::Communicator const& comm__, int const rank__);

    void
    sync_core_charge_density(mpi::Communicator const& comm__, int const rank__);

    /// Check if local orbitals are linearly independent
    std::vector<int>
    check_lo_linear_independence(double etol__) const;

    /// Dump local orbitals to the file for debug purposes
    void
    dump_lo();

    /// Find core states and generate core density.
    void
    generate_core_charge_density(relativity_t core_rel__);

    /// Find linearization energy.
    void
    find_enu(relativity_t rel__);

    void
    write_enu(mpi::pstdout& pout) const;

    /// Generate radial overlap and SO integrals
    /** In the case of spin-orbit interaction the following integrals are computed:
     *  \f[
     *      \int f_{p}(r) \Big( \frac{1}{(2 M c)^2} \frac{1}{r} \frac{d V}{d r} \Big) f_{p'}(r) r^2 dr
     *  \f]
     *
     *  Relativistic mass M is defined as
     *  \f[
     *      M = 1 - \frac{1}{2 c^2} V
     *  \f]
     */
    void
    generate_radial_integrals(relativity_t rel__);

    /// Get m-th order radial derivative of AW functions at the MT surface.
    inline double
    aw_surface_deriv(int l__, int order__, int dm__) const
    {
        RTE_ASSERT(dm__ <= 2);
        auto idxrf = atom_type_.indexr().index_of(angular_momentum(l__), order__);
        return surface_derivatives_(dm__, idxrf);
    }

    /// Set surface derivative of AW radial functions.
    inline void
    aw_surface_deriv(int l__, int order__, int dm__, double deriv__)
    {
        RTE_ASSERT(dm__ <= 2);
        auto idxrf                        = atom_type_.indexr().index_of(angular_momentum(l__), order__);
        surface_derivatives_(dm__, idxrf) = deriv__;
    }

    /// Return symmetry class id.
    inline int
    id() const
    {
        return id_;
    }

    /// Add atom id to the current class.
    inline void
    add_atom_id(int atom_id__)
    {
        atom_id_.push_back(atom_id__);
    }

    /// Return number of atoms belonging to the current symmetry class.
    inline int
    num_atoms() const
    {
        return static_cast<int>(atom_id_.size());
    }

    inline int
    atom_id(int idx) const
    {
        return atom_id_[idx];
    }

    /// Get a value of the radial functions.
    inline double
    radial_function(int ir, int idx) const
    {
        return radial_functions_(ir, idx, 0);
    }

    /// Set radial function.
    inline void
    radial_function(int idx__, std::vector<double> f__)
    {
        for (int ir = 0; ir < this->atom_type().num_mt_points(); ir++) {
            radial_functions_(ir, idx__, 0) = f__[ir];
        }
    }

    /// Set radial function derivative r*(du/dr).
    inline void
    radial_function_derivative(int idx__, std::vector<double> f__)
    {
        for (int ir = 0; ir < this->atom_type().num_mt_points(); ir++) {
            radial_functions_(ir, idx__, 1) = f__[ir];
        }
    }

    inline double
    h_spherical_integral(int i1, int i2) const
    {
        return h_spherical_integrals_(i1, i2);
    }

    inline double const&
    o_radial_integral(int l, int order1, int order2) const
    {
        return o_radial_integrals_(l, order1, order2);
    }

    inline void
    set_o_radial_integral(int l, int order1, int order2, double oint__)
    {
        o_radial_integrals_(l, order1, order2) = oint__;
    }

    inline double const&
    o1_radial_integral(int xi1__, int xi2__) const
    {
        return o1_radial_integrals_(xi1__, xi2__);
    }

    inline void
    set_o1_radial_integral(int idxrf1__, int idxrf2__, double val__)
    {
        o1_radial_integrals_(idxrf1__, idxrf2__) = val__;
    }

    inline double
    so_radial_integral(int l, int order1, int order2) const
    {
        return so_radial_integrals_(l, order1, order2);
    }

    inline double
    ae_core_charge_density(int ir) const
    {
        RTE_ASSERT(ir >= 0 && ir < (int)ae_core_charge_density_.size());

        return ae_core_charge_density_[ir];
    }

    inline Atom_type const&
    atom_type() const
    {
        return atom_type_;
    }

    inline double
    core_eval_sum() const
    {
        return core_eval_sum_;
    }

    inline double
    core_leakage() const
    {
        return core_leakage_;
    }

    inline int
    num_aw_descriptors() const
    {
        return static_cast<int>(aw_descriptors_.size());
    }

    inline radial_solution_descriptor_set&
    aw_descriptor(int idx__) const
    {
        return aw_descriptors_[idx__];
    }

    inline int
    num_lo_descriptors() const
    {
        return static_cast<int>(lo_descriptors_.size());
    }

    inline local_orbital_descriptor&
    lo_descriptor(int idx__) const
    {
        return lo_descriptors_[idx__];
    }

    inline void
    set_aw_enu(int l, int order, double enu)
    {
        aw_descriptors_[l][order].enu = enu;
    }

    inline double
    get_aw_enu(int l, int order) const
    {
        return aw_descriptors_[l][order].enu;
    }

    inline void
    set_lo_enu(int idxlo, int order, double enu)
    {
        lo_descriptors_[idxlo].rsd_set[order].enu = enu;
    }

    inline double
    get_lo_enu(int idxlo, int order) const
    {
        return lo_descriptors_[idxlo].rsd_set[order].enu;
    }
};

} // namespace sirius

#endif // __ATOM_SYMMETRY_CLASS_H__
