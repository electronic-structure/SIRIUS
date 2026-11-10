/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file lapw_radial_basis.hpp
 *
 *  \brief Radial basis functions of the LAPW method.
 */

#ifndef __LAPW_RADIAL_BASIS_HPP__
#define __LAPW_RADIAL_BASIS_HPP__

#include "unit_cell/unit_cell.hpp"
#include "core/la/eigensolver.hpp"
#include "radial/enu_search.hpp"

namespace sirius {

struct lapw_radial_basis_t
{
    /// Reference to atom type.
    Atom_type const& atom_type_;

    relativity_t rel_;

    /// Spherical part of the effective potential.
    /** Used by the LAPW radial solver. Actual value is stored, not the Y00 component. */
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

    std::map<std::pair<int, int>, enu_search_t> enu_search_;

    /// List of radial descriptor sets used to construct augmented waves.
    std::vector<radial_solution_descriptor_set> aw_descriptors_;

    /// List of radial descriptor sets used to construct local orbitals.
    std::vector<local_orbital_descriptor> lo_descriptors_;

    lapw_radial_basis_t(Atom_type const& atom_type__, relativity_t rel__, std::vector<double> spherical_potential__);

    /// Find linearization energy.
    int
    find_enu();

    int
    generate_aw_radial_functions();

    int
    generate_lo_radial_functions();

    /// Check if local orbitals are linearly independent
    std::vector<int>
    check_lo_linear_independence(double tol__) const;

    /// Generate APW and LO radial functions.
    int
    generate_radial_functions();

    void
    generate_h_spherical_integrals();

    void
    generate_o_radial_integrals();

    void
    generate_o1_radial_integrals();

    void
    generate_so_radial_integrals();

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
    generate_radial_integrals();

    inline double
    h_spherical_integral(int i1__, int i2__) const
    {
        return h_spherical_integrals_(i1__, i2__);
    }

    inline double
    o_radial_integral(int l__, int order1__, int order2__) const
    {
        return o_radial_integrals_(l__, order1__, order2__);
    }

    inline double
    o1_radial_integral(int i1__, int i2__) const
    {
        return o1_radial_integrals_(i1__, i2__);
    }

    inline double
    so_radial_integral(int l__, int order1__, int order2__) const
    {
        return so_radial_integrals_(l__, order1__, order2__);
    }

    /// Get m-th order radial derivative of AW functions at the MT surface.
    inline double
    aw_surface_deriv(int l__, int order__, int dm__) const;

    /// Get a value of the radial functions.
    inline double
    radial_function(int ir__, int idx__) const
    {
        return radial_functions_(ir__, idx__, 0);
    }

    template <typename T>
    inline void
    write_enu(T& pout) const;

    /// Save spherical potential to a json file (for debugging).
    void
    save_spherical_potential() const;

    /// Save radial functions to a json file (for debugging).
    void
    save_radial_functions(std::string const& fname__) const;
};

inline lapw_radial_basis_t::lapw_radial_basis_t(Atom_type const& atom_type__, relativity_t rel__,
                                                std::vector<double> spherical_potential__)
    : atom_type_(atom_type__)
    , rel_(rel__)
    , spherical_potential_(spherical_potential__)
{
    int nl        = atom_type_.indexr().lmax() + 1;
    int max_order = atom_type_.indexr().max_order();
    int nrf       = atom_type_.mt_radial_basis_size();

    radial_functions_ = mdarray<double, 3>({atom_type_.num_mt_points(), nrf, 2});
    radial_functions_.zero();

    surface_derivatives_ = mdarray<double, 2>({3, atom_type_.mt_radial_basis_size() - atom_type_.num_lo_descriptors()});
    surface_derivatives_.zero();

    h_spherical_integrals_ = mdarray<double, 2>({nrf, nrf});
    h_spherical_integrals_.zero();

    o_radial_integrals_ = mdarray<double, 3>({nl, max_order, max_order});
    o_radial_integrals_.zero();

    so_radial_integrals_ = mdarray<double, 3>({nl, max_order, max_order});
    so_radial_integrals_.zero();

    if (rel_ == relativity_t::iora) {
        o1_radial_integrals_ = mdarray<double, 2>({nrf, nrf});
        o1_radial_integrals_.zero();
    }

    /* copy descriptors because enu is different between atom classes */
    for (int i = 0; i < atom_type_.num_aw_descriptors(); i++) {
        aw_descriptors_.push_back(atom_type_.aw_descriptor(i));
    }

    for (int i = 0; i < atom_type_.num_lo_descriptors(); i++) {
        lo_descriptors_.push_back(atom_type_.lo_descriptor(i));
    }

    /* find which aw functions need auto enu */
    for (int l = 0; l < atom_type_.num_aw_descriptors(); l++) {
        for (auto const& d : aw_descriptors_[l]) {
            if (d.auto_enu) {
                enu_search_[{d.n, d.l}] = enu_search_t{d.enu, d.enu, d.enu, d.auto_enu};
            }
        }
    }

    /* find which lo functions need auto enu */
    for (int idxlo = 0; idxlo < atom_type_.num_lo_descriptors(); idxlo++) {
        for (auto const& d : lo_descriptors_[idxlo].rsd_set) {
            if (d.auto_enu) {
                enu_search_[{d.n, d.l}] = enu_search_t{d.enu, d.enu, d.enu, d.auto_enu};
            }
        }
    }
}

inline double
lapw_radial_basis_t::aw_surface_deriv(int l__, int order__, int dm__) const
{
    RTE_ASSERT(dm__ <= 2);
    auto idxrf = atom_type_.indexr().index_of(angular_momentum(l__), order__);
    return surface_derivatives_(dm__, idxrf);
}

template <typename T>
inline void
lapw_radial_basis_t::write_enu(T& pout) const
{
    pout << "Atom : " << atom_type_.symbol() << std::endl;
    pout << "augmented waves" << std::endl;
    for (int l = 0; l < atom_type_.num_aw_descriptors(); l++) {
        for (size_t order = 0; order < aw_descriptors_[l].size(); order++) {
            auto& rsd = aw_descriptors_[l][order];
            if (rsd.auto_enu) {
                pout << rsd << std::endl;
            }
        }
    }

    pout << "local orbitals" << std::endl;
    for (int idxlo = 0; idxlo < atom_type_.num_lo_descriptors(); idxlo++) {
        for (size_t order = 0; order < lo_descriptors_[idxlo].rsd_set.size(); order++) {
            auto& rsd = lo_descriptors_[idxlo].rsd_set[order];
            if (rsd.auto_enu) {
                pout << rsd << std::endl;
            }
        }
    }
    pout << std::endl;
}

class LAPW_radial_basis
{
  private:
    std::vector<lapw_radial_basis_t> radial_basis_of_symmetry_class;
    Unit_cell const& unit_cell_;

    void
    sync_radial_functions(int ic__, int rank__);

    void
    sync_radial_integrals(int ic__, int rank__);

  public:
    LAPW_radial_basis(Unit_cell const& unit_cell__, relativity_t rel__, std::vector<std::vector<double>> vs__);

    lapw_radial_basis_t const&
    radial_basis(typename atom_index_t::global ia__) const;

    void
    write_enu(std::ostream& out__) const;
};

inline void
LAPW_radial_basis::sync_radial_functions(int ic__, int rank__)
{
    auto& rb = radial_basis_of_symmetry_class[ic__];

    unit_cell_.comm().bcast(rb.radial_functions_.at(memory_t::host), static_cast<int>(rb.radial_functions_.size()),
                            rank__);
    unit_cell_.comm().bcast(rb.surface_derivatives_.at(memory_t::host),
                            static_cast<int>(rb.surface_derivatives_.size()), rank__);
}

inline LAPW_radial_basis::LAPW_radial_basis(Unit_cell const& unit_cell__, relativity_t rel__,
                                            std::vector<std::vector<double>> vs__)
    : unit_cell_{unit_cell__}
{
    for (int ic = 0; ic < unit_cell__.num_atom_symmetry_classes(); ic++) {
        radial_basis_of_symmetry_class.emplace_back(unit_cell__.atom_symmetry_class(ic).atom_type(), rel__, vs__[ic]);
    }

    auto spl_num_symcls = splindex_block<atom_symmetry_class_index_t>(unit_cell__.num_atom_symmetry_classes(),
                                                                      n_blocks(unit_cell__.comm().size()),
                                                                      block_id(unit_cell__.comm().rank()));

    int ierr{0};
    for (auto it : spl_num_symcls) {
        radial_basis_of_symmetry_class[it.i].find_enu();
        ierr += radial_basis_of_symmetry_class[it.i].generate_radial_functions();
    }
    unit_cell__.comm().allreduce<int, mpi::op_t::max>(&ierr, 1);
    if (ierr) {
        RTE_THROW("lapw radial basis was not generated");
    }
    for (auto it : spl_num_symcls) {
        radial_basis_of_symmetry_class[it.i].generate_radial_integrals();
    }

    for (int ic = 0; ic < unit_cell__.num_atom_symmetry_classes(); ic++) {
        int rank = spl_num_symcls.location(typename atom_symmetry_class_index_t::global(ic)).ib;
        sync_radial_functions(ic, rank);
        sync_radial_integrals(ic, rank);
    }
}

inline lapw_radial_basis_t const&
LAPW_radial_basis::radial_basis(typename atom_index_t::global ia__) const
{
    int icls = unit_cell_.atom(ia__).symmetry_class().id();
    return radial_basis_of_symmetry_class[icls];
}

} // namespace sirius

#endif
