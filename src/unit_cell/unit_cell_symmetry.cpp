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

/** \file unit_cell_symmetry.cpp
 *
 *  \brief Contains implementation of sirius::Unit_cell_symmetry class.
 */

#include "unit_cell_symmetry.hpp"

using namespace geometry3d;

namespace sirius {

static std::vector<int>
find_sym_atom(int num_atoms__, sddk::mdarray<double, 2> const& positions__, matrix3d<int> const& R__,
              vector3d<double> const& t__, double tolerance__)
{
    PROFILE("sirius::find_sym_atom");

    std::vector<int> sym_atom(num_atoms__);

    auto distance = [](const vector3d<double>& a, const vector3d<double>& b)
    {
        auto diff = a - b;
        for (int x: {0, 1, 2}) {
            double dl = std::abs(diff[x]);
            diff[x] = std::min(dl, 1 - dl);
        }
        return diff.length();
    };

    for (int ia = 0; ia < num_atoms__; ia++) {
        /* spatial transform */
        vector3d<double> pos(positions__(0, ia), positions__(1, ia), positions__(2, ia));
        /* apply crystal symmetry */
        auto v = reduce_coordinates(dot(R__, pos) + t__);
        double d0{1e10};
        double j0{-1};
        vector3d<double> p0;

        int ja{-1};
        /* check for equivalent atom; remember that the atomic positions are not necessarily in [0,1) interval
           and the reduction of coordinates is required */
        for (int k = 0; k < num_atoms__; k++) {
            vector3d<double> pos1(positions__(0, k), positions__(1, k), positions__(2, k));
            /* find the distance between original and trasformed atoms */
            double dist = distance(v.first, reduce_coordinates(pos1).first);
            if (dist < tolerance__) {
                ja = k;
                break;
            }
            if (dist < d0) {
                d0 = dist;
                j0 = k;
                p0 = pos1;
            }
        }

        if (ja == -1) {
            std::stringstream s;
            s << "equivalent atom was not found\n"
              << "  initial atom: " << ia << ", position : " << pos << ", reduced: " << v.first << "\n"
              << "  nearest atom: " << j0 << ", position : " << p0 << ", reduced: " << reduce_coordinates(p0).first << "\n"
              << "  distance between atoms: " << d0 << "\n"
              << "  tolerance: " << tolerance__;
            TERMINATE(s);
        }
        sym_atom[ia] = ja;
    }
    return sym_atom;
}

static space_group_symmetry_descriptor
get_spg_sym_op(int isym_spg__, SpglibDataset* spg_dataset__, matrix3d<double> const& lattice_vectors__, int num_atoms__,
    sddk::mdarray<double, 2> const& positions__, double tolerance__)
{
    space_group_symmetry_descriptor sym_op;

    auto inverse_lattice_vectors = inverse(lattice_vectors__);

    /* rotation matrix in lattice coordinates */
    sym_op.R = matrix3d<int>(spg_dataset__->rotations[isym_spg__]);
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
    sym_op.t = vector3d<double>(spg_dataset__->translations[isym_spg__][0],
                                spg_dataset__->translations[isym_spg__][1],
                                spg_dataset__->translations[isym_spg__][2]);
    /* is this proper or improper rotation */
    sym_op.proper = p;
    /* proper rotation in cartesian Coordinates */
    sym_op.rotation = dot(dot(lattice_vectors__, matrix3d<double>(sym_op.R * p)), inverse_lattice_vectors);
    /* get Euler angles of the rotation */
    sym_op.euler_angles = euler_angles(sym_op.rotation);
    /* get symmetry related atoms */
    sym_op.sym_atom = find_sym_atom(num_atoms__, positions__, sym_op.R, sym_op.t, tolerance__);

    return sym_op;
}

static space_group_symmetry_descriptor
get_identity_spg_sym_op(int num_atoms__)
{
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
    sym_op.sym_atom = std::vector<int>(num_atoms__);
    std::iota(sym_op.sym_atom.begin(), sym_op.sym_atom.end(), 0);

    return sym_op;
}

Unit_cell_symmetry::Unit_cell_symmetry(matrix3d<double> const& lattice_vectors__, int num_atoms__,
    std::vector<int> const& types__, sddk::mdarray<double, 2> const& positions__,
    sddk::mdarray<double, 2> const& spins__, bool spin_orbit__, double tolerance__, bool use_sym__)
    : lattice_vectors_(lattice_vectors__)
    , num_atoms_(num_atoms__)
    , types_(types__)
    , tolerance_(tolerance__)
{
    PROFILE("sirius::Unit_cell_symmetry");

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

    PROFILE_START("sirius::Unit_cell_symmetry|spg");
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
    PROFILE_STOP("sirius::Unit_cell_symmetry|spg");

    if (spg_dataset_) {
        /* make a list of crystal symmetries */
        for (int isym = 0; isym < spg_dataset_->n_operations; isym++) {
            auto sym_op = get_spg_sym_op(isym, spg_dataset_, lattice_vectors__, num_atoms__, positions__, tolerance__);
            /* add symmetry operation to a list */
            space_group_symmetry_.push_back(sym_op);
        }
    } else { /* add only identity element */
        auto sym_op = get_identity_spg_sym_op(num_atoms__);
        /* add symmetry operation to a list */
        space_group_symmetry_.push_back(sym_op);
    }

    PROFILE_START("sirius::Unit_cell_symmetry|mag");
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
            /* check if all atoms transform under spatial and spin symmetries */
            for (int ia = 0; ia < num_atoms_; ia++) {
                int ja = space_group_symmetry(isym).sym_atom[ia];

                /* now check that vector field transforms from atom ia to atom ja */
                /* vector field of atom is expected to be in Cartesian coordinates */
                auto vd = dot(Rspin, vector3d<double>(spins__(0, ia), spins__(1, ia), spins__(2, ia))) -
                                  vector3d<double>(spins__(0, ja), spins__(1, ja), spins__(2, ja));

                if (vd.length() < 1e-10) {
                    n++;
                }
            }
            /* if all atoms transform under spin rotaion, add it to a list */
            if (n == num_atoms_) {
                magnetic_group_symmetry_descriptor mag_op;
                mag_op.spg_op            = space_group_symmetry(isym);
                mag_op.spin_rotation     = Rspin;
                mag_op.spin_rotation_inv = inverse(Rspin);
                mag_op.spin_rotation_su2 = rotation_matrix_su2(Rspin);
                /* add symmetry to the list */
                magnetic_group_symmetry_.push_back(std::move(mag_op));
                break;
            }
        }
    }
    PROFILE_STOP("sirius::Unit_cell_symmetry|mag");
}

} // namespace

