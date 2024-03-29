/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file crystal_symmetry.hpp
 *
 *  \brief Contains definition and partial implementation of sirius::Crystal_symmetry class.
 */

#ifndef __CRYSTAL_SYMMETRY_HPP__
#define __CRYSTAL_SYMMETRY_HPP__

#include <cstddef>

extern "C" {
#include <spglib.h>
}

#include "core/memory.hpp"
#include "core/r3/r3.hpp"
#include "core/profiler.hpp"

namespace sirius {

/// Descriptor of the space group symmetry operation.
struct space_group_symmetry_descriptor
{
    /// Rotational part of symmetry operation (fractional coordinates).
    r3::matrix<int> R;

    /// Inverse of R.
    r3::matrix<int> invR;

    /// Inverse transposed of R.
    r3::matrix<int> invRT;

    /// Proper rotation matrix in Cartesian coordinates.
    r3::matrix<double> Rcp;

    /// (Im)proper Rotation matrix in Cartesian coordinates.
    r3::matrix<double> Rc;

    /// Fractional translation.
    r3::vector<double> t;

    /// Proper (+1) or improper (-1) rotation.
    int proper;

    /// Three Euler angles that generate the proper rotation matrix.
    r3::vector<double> euler_angles;

    /// Symmetry table.
    std::vector<int> sym_atom;

    /// Invere symmetry table.
    std::vector<int> inv_sym_atom;

    /// Translation vector that prings symmetry-transformed atom back to the unit cell.
    std::vector<r3::vector<int>> inv_sym_atom_T;
};

/// Descriptor of the magnetic group symmetry operation.
struct magnetic_group_symmetry_descriptor
{
    /// Element of space group symmetry.
    space_group_symmetry_descriptor spg_op;

    /// Proper rotation matrix in Cartesian coordinates.
    r3::matrix<double> spin_rotation;

    /// Inverse of proper spin rotation matrix in Cartesian coordinates.
    r3::matrix<double> spin_rotation_inv;

    mdarray<std::complex<double>, 2> spin_rotation_su2;
};

/// Representation of the crystal symmetry.
class Crystal_symmetry
{
  private:
    /// Matrix of lattice vectors.
    /** Spglib requires this matrix to have a positively defined determinant. */
    r3::matrix<double> lattice_vectors_;

    /// Inverse of the lattice vectors matrix.
    r3::matrix<double> inverse_lattice_vectors_;

    /// Number of atoms in the unit cell.
    int num_atoms_;

    /// Number of atom types.
    int num_atom_types_;

    /// Atom types.
    std::vector<int> types_;

    /// Atomic positions.
    mdarray<double, 2> positions_;

    /// Magnetic moments of atoms.
    mdarray<double, 2> magnetization_;

    double tolerance_;

    /// Crystal structure descriptor returned by spglib.
    SpglibDataset* spg_dataset_{nullptr};

    /// List of all space group symmetry operations.
    std::vector<space_group_symmetry_descriptor> space_group_symmetry_;

    /// List of all magnetic group symmetry operations.
    std::vector<magnetic_group_symmetry_descriptor> magnetic_group_symmetry_;

    /// Number of crystal symmetries without magnetic configuration.
    inline int
    num_spg_sym() const
    {
        return static_cast<int>(space_group_symmetry_.size());
    }

    inline auto const&
    space_group_symmetry(int isym__) const
    {
        assert(isym__ >= 0 && isym__ < num_spg_sym());
        return space_group_symmetry_[isym__];
    }

  public:
    Crystal_symmetry(r3::matrix<double> const& lattice_vectors__, int num_atoms__, int num_atom_types__,
                     std::vector<int> const& types__, mdarray<double, 2> const& positions__,
                     mdarray<double, 2> const& spins__, bool spin_orbit__, double tolerance__, bool use_sym__);

    ~Crystal_symmetry()
    {
        if (spg_dataset_) {
            spg_free_dataset(spg_dataset_);
        }
    }

    inline int
    atom_symmetry_class(int ia__) const
    {
        if (spg_dataset_) {
            return spg_dataset_->equivalent_atoms[ia__];
        } else {
            return ia__;
        }
    }

    inline int
    spacegroup_number() const
    {
        if (spg_dataset_) {
            return spg_dataset_->spacegroup_number;
        } else {
            return 0;
        }
    }

    inline auto
    international_symbol() const
    {
        if (spg_dataset_) {
            return std::string(spg_dataset_->international_symbol);
        } else {
            return std::string("n/a");
        }
    }

    inline auto
    hall_symbol() const
    {
        if (spg_dataset_) {
            return std::string(spg_dataset_->hall_symbol);
        } else {
            return std::string("n/a");
        }
    }

    inline auto
    transformation_matrix() const
    {
        if (spg_dataset_) {
            return r3::matrix<double>(spg_dataset_->transformation_matrix);
        } else {
            return r3::matrix<double>({{1.0, 0, 0}, {0, 1.0, 0}, {0, 0, 1.0}});
        }
    }

    inline auto
    origin_shift() const
    {
        if (spg_dataset_) {
            return r3::vector<double>(spg_dataset_->origin_shift[0], spg_dataset_->origin_shift[1],
                                      spg_dataset_->origin_shift[2]);
        } else {
            return r3::vector<double>(0, 0, 0);
        }
    }

    /// Number of symmetries including the magnetic configuration.
    /** This is less or equal to the number of crystal symmetries. */
    inline int
    size() const
    {
        return static_cast<int>(magnetic_group_symmetry_.size());
    }

    inline auto const&
    operator[](int isym__) const
    {
        assert(isym__ >= 0 && isym__ < this->size());
        return magnetic_group_symmetry_[isym__];
    }

    auto const&
    lattice_vectors() const
    {
        return lattice_vectors_;
    }

    auto const&
    inverse_lattice_vectors() const
    {
        return inverse_lattice_vectors_;
    }

    inline auto
    num_atoms() const
    {
        return num_atoms_;
    }

    inline auto
    num_atom_types() const
    {
        return num_atom_types_;
    }

    inline auto
    atom_type(int ia__) const
    {
        return types_[ia__];
    }

    /// Get an error in metric tensor.
    /** Metric tensor in transformed under lattice symmetry operations and compareed with
     *  the initial value. It should stay invariant under transformation. This, however,
     *  is not always guaranteed numerically, especially when spglib uses large tolerance
     *  and find more symmeetry operations.
     *
     *  The error is the maximum value of \f$ |M_{ij} - \tilde M_{ij}| \f$ where \f$ M_{ij} \f$
     *  is the initial metric tensor and \f$ \tilde M_{ij} \f$ is the transformed tensor. */
    double
    metric_tensor_error() const;

    /// Get error in rotation matrix of the symmetry operation.
    /** Comparte rotation matrix in Cartesian coordinates with its inverse transpose. They should match.
     *
     *  The error is the maximum value of \f$ |R_{ij} - R_{ij}^{-T}| \f$, where \f$ R_{ij} \f$ is the rotation
     *  matrix and \f$  R_{ij}^{-T} \f$ inverse transpose of the rotation matrix. */
    double
    sym_op_R_error() const;

    /// Print information about the unit cell symmetry.
    void
    print_info(std::ostream& out__, int verbosity__) const;
};

} // namespace sirius

/** \page sym Symmetry
 *  \section section1 Definition of symmetry operation
 *
 *  SIRIUS uses Spglib to find the spacial symmetry operations. Spglib defines symmetry operation in fractional
 *  coordinates:
 *  \f[
 *      {\bf x'} = \{ {\bf R} | {\bf t} \} {\bf x} \equiv {\bf R}{\bf x} + {\bf t}
 *  \f]
 *  where \b R is the proper or improper rotation matrix with elements equal to -1,0,1 and determinant of 1
 *  (pure rotation) or -1 (rotoreflection) and \b t is the fractional translation, associated with the symmetry
 *  operation. The inverse of the symmetry operation is:
 *  \f[
 *      {\bf x} = \{ {\bf R} | {\bf t} \}^{-1} {\bf x'} = {\bf R}^{-1} ({\bf x'} - {\bf t}) =
 *          {\bf R}^{-1} {\bf x'} - {\bf R}^{-1} {\bf t}
 *  \f]
 *
 *  We will always use an \a active transformation (transformation of vectors or functions) and never a passive
 *  transformation (transformation of coordinate system). However one should remember definition of the function
 *  transformation:
 *  \f[
 *      \hat {\bf P} f({\bf r}) \equiv f(\hat {\bf P}^{-1} {\bf r})
 *  \f]
 *
 *  It is straightforward to get the rotation matrix in Cartesian coordinates. We know how the vector in Cartesian
 *  coordinates is obtained from the vector in fractional coordinates:
 *  \f[
 *      {\bf v} = {\bf L} {\bf x}
 *  \f]
 *  where \b L is the 3x3 matrix which clomuns are three lattice vectors. The backward transformation is simply
 *  \f[
 *      {\bf x} = {\bf L}^{-1} {\bf v}
 *  \f]
 *  Now we write rotation operation in fractional coordinates and apply the backward transformation to Cartesian
 *  coordinates:
 *  \f[
 *      {\bf x'} = {\bf R}{\bf x} \rightarrow {\bf L}^{-1} {\bf v'} = {\bf R} {\bf L}^{-1} {\bf v}
 *  \f]
 *  from which we derive the rotation operation in Cartesian coordinates:
 *  \f[
 *      {\bf v'} = {\bf L} {\bf R} {\bf L}^{-1} {\bf v}
 *  \f]
 */

#endif // __UNIT_CELL_SYMMETRY_H__
