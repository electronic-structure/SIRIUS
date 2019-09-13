// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file unit_cell_symmetry.hpp
 *
 *  \brief Contains definition and implementation of sirius::Unit_cell_symmetry class.
 */

#ifndef __UNIT_CELL_SYMMETRY_HPP__
#define __UNIT_CELL_SYMMETRY_HPP__

extern "C" {
#include <spglib.h>
}

//#include "constants.hpp"
#include "Symmetry/rotation.hpp"
#include "utils/profiler.hpp"

using namespace geometry3d;

namespace sirius {

/// Descriptor of the space group symmetry operation.
struct space_group_symmetry_descriptor
{
    /// Rotational part of symmetry operation (fractional coordinates).
    matrix3d<int> R;

    /// Inverse of R.
    matrix3d<int> invR;

    /// Inverse transposed of R.
    matrix3d<int> invRT;

    /// Fractional translation.
    vector3d<double> t;

    /// Proper (+1) or improper (-1) rotation.
    int proper;

    /// Proper rotation matrix in Cartesian coordinates.
    matrix3d<double> rotation;

    /// Three Euler angles that generate the proper rotation matrix.
    vector3d<double> euler_angles;
};

/// Descriptor of the magnetic group symmetry operation.
struct magnetic_group_symmetry_descriptor
{
    /// Element of space group symmetry.
    space_group_symmetry_descriptor spg_op;

    /// Index of the space group symmetry operation.
    /** This index is used to search for the transfomation of atoms under the current space group operation
     *  in the precomputed symmetry table. */
    int isym;

    /// Proper rotation matrix in Cartesian coordinates.
    matrix3d<double> spin_rotation;

    /// Inverse of proper spin rotation matrix in Cartesian coordinates.
    matrix3d<double> spin_rotation_inv;
};

/// Representation of the unit cell symmetry.
class Unit_cell_symmetry
{
  private:

    /// Matrix of lattice vectors.
    /** Spglib requires this matrix to have a positively defined determinant. */
    matrix3d<double> lattice_vectors_;

    /// Inverse of the lattice vectors matrix.
    matrix3d<double> inverse_lattice_vectors_;

    /// Number of atoms in the unit cell.
    int num_atoms_;

    /// Atom types.
    std::vector<int> types_;

    /// Atomic positions.
    mdarray<double, 2> positions_;

    /// Magnetic moments of atoms.
    mdarray<double, 2> magnetization_;

    double tolerance_;

    /// Crystal structure descriptor returned by spglib.
    SpglibDataset* spg_dataset_{nullptr};

    /// Symmetry table for atoms.
    /** For each atom ia and symmetry isym sym_table_(ia, isym) stores index of atom ja to which original atom
     *  transforms under symmetry operation. */
    mdarray<int, 2> sym_table_;

    /// List of all space group symmetry operations.
    std::vector<space_group_symmetry_descriptor> space_group_symmetry_;

    /// List of all magnetic group symmetry operations.
    std::vector<magnetic_group_symmetry_descriptor> magnetic_group_symmetry_;

  public:

    Unit_cell_symmetry(matrix3d<double> const& lattice_vectors__, int num_atoms__, std::vector<int> const& types__,
                       mdarray<double, 2> const& positions__, mdarray<double, 2> const& spins__, bool spin_orbit__,
                       double tolerance__, bool use_sym__)
        : lattice_vectors_(lattice_vectors__)
        , num_atoms_(num_atoms__)
        , types_(types__)
        , tolerance_(tolerance__)
    {
        PROFILE("sirius::Unit_cell_symmetry::Unit_cell_symmetry");

        /* check lattice vectors */
        if (lattice_vectors__.det() < 0 && use_sym__) {
            std::stringstream s;
            s << "spglib requires positive determinant for a matrix of lattice vectors";
            TERMINATE(s);
        }

        /* make inverse */
        inverse_lattice_vectors_ = inverse(lattice_vectors_);

        double lattice[3][3];
        for (int i: {0, 1, 2}) {
            for (int j: {0, 1, 2}) {
                lattice[i][j] = lattice_vectors_(i, j);
            }
        }

        positions_ = mdarray<double, 2>(3, num_atoms_);
        positions__ >> positions_;

        magnetization_ = mdarray<double, 2>(3, num_atoms_);
        spins__ >> magnetization_;

        utils::timer t1("sirius::Unit_cell_symmetry|spg");
        if (use_sym__) {
            spg_dataset_ = spg_get_dataset(lattice, (double(*)[3])&positions_(0, 0), &types_[0], num_atoms_, tolerance_);
            if (spg_dataset_ == NULL) {
                TERMINATE("spg_get_dataset() returned NULL");
            }

            if (spg_dataset_->spacegroup_number == 0) {
                TERMINATE("spg_get_dataset() returned 0 for the space group");
            }

            if (spg_dataset_->n_atoms != num_atoms__) {
                std::stringstream s;
                s << "spg_get_dataset() returned wrong number of atoms (" << spg_dataset_->n_atoms << ")" << std::endl
                  << "expected number of atoms is " <<  num_atoms__;
                TERMINATE(s);
            }
        }
        t1.stop();

        if (spg_dataset_) {
            /* make a list of crystal symmetries */
            for (int isym = 0; isym < spg_dataset_->n_operations; isym++) {
                space_group_symmetry_descriptor sym_op;

                /* rotation matrix in lattice coordinates */
                sym_op.R = matrix3d<int>(spg_dataset_->rotations[isym]);
                /* sanity check */
                int p = sym_op.R.det();
                if (!(p == 1 || p == -1)) {
                    TERMINATE("wrong rotation matrix");
                }
                /* inverse of the rotation matrix */
                sym_op.invR = inverse(sym_op.R);
                /* inverse transpose */
                sym_op.invRT = transpose(sym_op.invR);
                /* fractional translation */
                sym_op.t = vector3d<double>(spg_dataset_->translations[isym][0],
                                            spg_dataset_->translations[isym][1],
                                            spg_dataset_->translations[isym][2]);
                /* is this proper or improper rotation */
                sym_op.proper = p;
                /* proper rotation in cartesian Coordinates */
                sym_op.rotation = lattice_vectors_ * matrix3d<double>(sym_op.R * p) * inverse_lattice_vectors_;
                /* get Euler angles of the rotation */
                sym_op.euler_angles = euler_angles(sym_op.rotation);
                /* add symmetry operation to a list */
                space_group_symmetry_.push_back(sym_op);
            }
        } else {
            space_group_symmetry_descriptor sym_op;
            sym_op.R = matrix3d<int>({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
            /* inverse of the rotation matrix */
            sym_op.invR = inverse(sym_op.R);
            /* inverse transpose */
            sym_op.invRT = transpose(sym_op.invR);
            /* fractional translation */
            sym_op.t = vector3d<double>(0, 0, 0);
            /* is this proper or improper rotation */
            sym_op.proper = 1;
            /* proper rotation in cartesian Coordinates */
            sym_op.rotation = matrix3d<double>({{1.0, 0, 0}, {0, 1.0, 0}, {0, 0, 1.0}});
            /* get Euler angles of the rotation */
            sym_op.euler_angles = euler_angles(sym_op.rotation);
            /* add symmetry operation to a list */
            space_group_symmetry_.push_back(sym_op);
        }

        utils::timer t3("sirius::Unit_cell_symmetry::Unit_cell_symmetry|sym2");
        sym_table_ = mdarray<int, 2>(num_atoms_, num_spg_sym());
        /* loop over spatial symmetries */
        #pragma omp parallel for schedule(static)
        for (int isym = 0; isym < num_spg_sym(); isym++) {
            for (int ia = 0; ia < num_atoms_; ia++) {
                auto R = space_group_symmetry(isym).R;
                auto t = space_group_symmetry(isym).t;
                /* spatial transform */
                vector3d<double> pos(positions__(0, ia), positions__(1, ia), positions__(2, ia));
                /* apply crystal symmetry */
                auto v = reduce_coordinates(R * pos + t);
                auto distance = [](const vector3d<double>& a, const vector3d<double>& b)
                {
                    auto diff = a - b;
                    for (int x: {0, 1, 2}) {
                        double dl = std::abs(diff[x]);
                        diff[x] = std::min(dl, 1 - dl);
                    }
                    return diff.length();
                };

                int ja{-1};
                /* check for equivalent atom */
                for (int k = 0; k < num_atoms_; k++) {
                    vector3d<double> pos1(positions__(0, k), positions__(1, k), positions__(2, k));
                    if (distance(v.first, pos1) < tolerance_) {
                        ja = k;
                        break;
                    }
                }

                if (ja == -1) {
                    TERMINATE("equivalent atom was not found");
                }
                sym_table_(ia, isym) = ja;
            }
        }
        t3.stop();

        utils::timer t4("sirius::Unit_cell_symmetry::Unit_cell_symmetry|sym3");
        /* loop over spatial symmetries */
        for (int isym = 0; isym < num_spg_sym(); isym++) {
            int jsym0 = 0;
            int jsym1 = num_spg_sym() - 1;
            if (spin_orbit__) {
                jsym0 = jsym1 = isym;
            }
            /* loop over spin symmetries */
            for (int jsym = jsym0; jsym <= jsym1; jsym++) {
                /* take proper part of rotation matrix */
                auto Rspin = space_group_symmetry(jsym).rotation;

                int n{0};
                /* check if all atoms transfrom under spatial and spin symmetries */
                for (int ia = 0; ia < num_atoms_; ia++) {
                    int ja = sym_table_(ia, isym);

                    /* now check that vector field transforms from atom ia to atom ja */
                    /* vector field of atom is expected to be in Cartesian coordinates */
                    auto vd = Rspin * vector3d<double>(spins__(0, ia), spins__(1, ia), spins__(2, ia)) -
                                      vector3d<double>(spins__(0, ja), spins__(1, ja), spins__(2, ja));

                    if (vd.length() < 1e-10) {
                        n++;
                    }
                }
                /* if all atoms transform under spin rotaion, add it to a list */
                if (n == num_atoms_) {
                    magnetic_group_symmetry_descriptor mag_op;
                    mag_op.spg_op        = space_group_symmetry(isym);
                    mag_op.isym          = isym;
                    mag_op.spin_rotation = Rspin;
                    mag_op.spin_rotation_inv = inverse(Rspin);
                    magnetic_group_symmetry_.push_back(mag_op);
                    break;
                }
            }
        }
        t4.stop();
    }

    ~Unit_cell_symmetry()
    {
        if (spg_dataset_) {
            spg_free_dataset(spg_dataset_);
        }
    }

    inline int atom_symmetry_class(int ia__)
    {
        if (spg_dataset_) {
            return spg_dataset_->equivalent_atoms[ia__];
        } else {
            return ia__;
        }
    }

    inline int spacegroup_number()
    {
        if (spg_dataset_) {
            return spg_dataset_->spacegroup_number;
        } else {
            return 0;
        }
    }

    inline std::string international_symbol()
    {
        if (spg_dataset_) {
            return spg_dataset_->international_symbol;
        } else {
            return std::string("n/a");
        }
    }

    inline std::string hall_symbol()
    {
        if (spg_dataset_) {
            return spg_dataset_->hall_symbol;
        } else {
            return std::string("n/a");
        }
    }

    matrix3d<double> transformation_matrix() const
    {
        if (spg_dataset_) {
            return matrix3d<double>(spg_dataset_->transformation_matrix);
        } else {
            return matrix3d<double>({{1.0, 0, 0}, {0, 1.0, 0}, {0, 0, 1.0}});
        }
    }

    vector3d<double> origin_shift() const
    {
        if (spg_dataset_) {
            return vector3d<double>(spg_dataset_->origin_shift[0],
                                    spg_dataset_->origin_shift[1],
                                    spg_dataset_->origin_shift[2]);
        } else {
            return vector3d<double>(0, 0, 0);
        }
    }

    /// Number of crystal symmetries without magnetic configuration.
    inline int num_spg_sym() const
    {
        return static_cast<int>(space_group_symmetry_.size());
    }

    inline space_group_symmetry_descriptor const& space_group_symmetry(int isym__) const
    {
        assert(isym__ >= 0 && isym__ < num_spg_sym());
        return space_group_symmetry_[isym__];
    }

    /// Number of symmetries including the magnetic configuration.
    /** This is less or equal to the number of crystal symmetries. */
    inline int num_mag_sym() const
    {
        return static_cast<int>(magnetic_group_symmetry_.size());
    }

    inline magnetic_group_symmetry_descriptor const& magnetic_group_symmetry(int isym__) const
    {
        assert(isym__ >= 0 && isym__ < num_mag_sym());
        return magnetic_group_symmetry_[isym__];
    }

    inline int sym_table(int ia__, int isym__) const
    {
        return sym_table_(ia__, isym__);
    }

    matrix3d<double> const& lattice_vectors() const
    {
        return lattice_vectors_;
    }

    matrix3d<double> const& inverse_lattice_vectors() const
    {
        return inverse_lattice_vectors_;
    }

    inline int num_atoms() const
    {
        return num_atoms_;
    }
};

} // namespace

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
