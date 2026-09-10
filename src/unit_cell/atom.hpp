/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file atom.hpp
 *
 *  \brief Contains declaration and partial implementation of sirius::Atom class.
 */

#ifndef __ATOM_HPP__
#define __ATOM_HPP__

#include "core/sht/gaunt.hpp"
#include "core/profiler.hpp"
#include "atom_symmetry_class.hpp"
#include "function3d/spheric_function.hpp"

namespace sirius {

/// Data and methods specific to the actual atom in the unit cell.
class Atom
{
  private:
    /// Type of the given atom.
    Atom_type const& type_;

    /// Symmetry class of the given atom.
    std::shared_ptr<Atom_symmetry_class> symmetry_class_;

    /// Position in fractional coordinates.
    r3::vector<double> position_;

    /// Vector field associated with the current site.
    r3::vector<double> vector_field_;

  public:
    /// Constructor.
    Atom(Atom_type const& type__, r3::vector<double> position__, r3::vector<double> vector_field__)
        : type_(type__)
        , position_(position__)
        , vector_field_(vector_field__)
    {
    }

    /// Return const reference to corresponding atom type object.
    inline auto const&
    type() const
    {
        return type_;
    }

    /// Return const referenced to atom symmetry class.
    inline auto const&
    symmetry_class() const
    {
        return *symmetry_class_;
    }

    /// Return atom type id.
    inline int
    type_id() const
    {
        return type_.id();
    }

    /// Return atom position in fractional coordinates.
    inline auto const&
    position() const
    {
        return position_;
    }

    /// Set atom position in fractional coordinates.
    inline void
    set_position(r3::vector<double> position__)
    {
        position_ = position__;
    }

    /// Return vector field.
    inline auto
    vector_field() const
    {
        return vector_field_;
    }

    /// Set vector field.
    inline void
    set_vector_field(r3::vector<double> vector_field__)
    {
        vector_field_ = vector_field__;
    }

    /// Return id of the symmetry class.
    inline int
    symmetry_class_id() const
    {
        if (symmetry_class_ != nullptr) {
            return symmetry_class_->id();
        }
        return -1;
    }

    /// Set symmetry class of the atom.
    inline void
    set_symmetry_class(std::shared_ptr<Atom_symmetry_class> symmetry_class__)
    {
        symmetry_class_ = std::move(symmetry_class__);
    }

    inline int
    num_mt_points() const
    {
        return type_.num_mt_points();
    }

    inline Radial_grid<double> const&
    radial_grid() const
    {
        return type_.radial_grid();
    }

    inline double
    radial_grid(int idx) const
    {
        return type_.radial_grid(idx);
    }

    inline double
    mt_radius() const
    {
        return type_.mt_radius();
    }

    inline int
    zn() const
    {
        return type_.zn();
    }

    inline int
    mt_basis_size() const
    {
        return type_.mt_basis_size();
    }

    inline int
    mt_aw_basis_size() const
    {
        return type_.mt_aw_basis_size();
    }

    inline int
    mt_lo_basis_size() const
    {
        return type_.mt_lo_basis_size();
    }
};

} // namespace sirius

#endif // __ATOM_H__
