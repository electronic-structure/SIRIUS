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

    /// Reference to atom type.
    Atom_type const& atom_type_;

  public:
    /// Constructor
    Atom_symmetry_class(int id__, Atom_type const& atom_type__)
        : id_(id__)
        , atom_type_(atom_type__)
    {
        if (!atom_type_.initialized()) {
            RTE_THROW("atom type is not initialized");
        }
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

    inline Atom_type const&
    atom_type() const
    {
        return atom_type_;
    }
};

} // namespace sirius

#endif // __ATOM_SYMMETRY_CLASS_H__
