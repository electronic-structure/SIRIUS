/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file crystal_symmetry.cpp
 *
 *  \brief Contains implementation of sirius::Crystal_symmetry class.
 */

#include <numeric>
#include "crystal_symmetry.hpp"
#include "lattice.hpp"
#include "rotation.hpp"
#include "core/ostream_tools.hpp"

namespace sirius {

static std::pair<std::vector<int>, std::vector<r3::vector<int>>>
find_sym_atom(int num_atoms__, mdarray<double, 2> const& positions__, r3::matrix<int> const& R__,
              r3::vector<double> const& t__, double tolerance__, bool inverse__ = false)
{
    PROFILE("sirius::find_sym_atom");

    std::vector<int> sym_atom(num_atoms__);
    std::vector<r3::vector<int>> sym_atom_T(num_atoms__);

    for (int ia = 0; ia < num_atoms__; ia++) {
        /* position of atom ia */
        r3::vector<double> r_ia(positions__(0, ia), positions__(1, ia), positions__(2, ia));
        /* apply crystal symmetry {R|t} or it's inverse */
        /* rp = {R|t}r or {R|t}^{-1}r */
        auto rp_ia = (inverse__) ? dot(inverse(R__), r_ia - t__) : dot(R__, r_ia) + t__;

        bool found{false};
        /* find the transformed atom */
        for (int ja = 0; ja < num_atoms__; ja++) {
            r3::vector<double> r_ja(positions__(0, ja), positions__(1, ja), positions__(2, ja));

            auto v = rp_ia - r_ja;

            r3::vector<int> T;
            double diff{0};
            for (int x : {0, 1, 2}) {
                T[x] = std::round(v[x]);
                diff += std::abs(T[x] - v[x]);
            }
            /* translation vector rp_ia = r_ja + T is found */
            if (diff < tolerance__) {
                found          = true;
                sym_atom[ia]   = ja;
                sym_atom_T[ia] = T;
                break;
            }
        }
        if (!found) {
            std::stringstream s;
            s << "equivalent atom was not found" << std::endl
              << "  symmetry operaton R:" << R__ << ", t: " << t__ << std::endl
              << "  atom: " << ia << ", position: " << r_ia << ", transformed position: " << rp_ia << std::endl
              << "  tolerance: " << tolerance__;
            RTE_THROW(s);
        }
    }
    return std::make_pair(sym_atom, sym_atom_T);
}

static space_group_symmetry_descriptor
get_spg_sym_op(int isym_spg__, SpglibDataset* spg_dataset__, r3::matrix<double> const& lattice_vectors__,
               int num_atoms__, mdarray<double, 2> const& positions__, double tolerance__)
{
    space_group_symmetry_descriptor sym_op;

    auto inverse_lattice_vectors = inverse(lattice_vectors__);

    /* rotation matrix in lattice coordinates */
    sym_op.R = r3::matrix<int>(spg_dataset__->rotations[isym_spg__]);
    /* sanity check */
    int p = sym_op.R.det();
    if (!(p == 1 || p == -1)) {
        RTE_THROW("wrong rotation matrix");
    }
    /* inverse of the rotation matrix */
    sym_op.invR = inverse(sym_op.R);
    /* inverse transpose */
    sym_op.invRT = transpose(sym_op.invR);
    /* fractional translation */
    sym_op.t =
            r3::vector<double>(spg_dataset__->translations[isym_spg__][0], spg_dataset__->translations[isym_spg__][1],
                               spg_dataset__->translations[isym_spg__][2]);
    /* is this proper or improper rotation */
    sym_op.proper = p;
    /* rotation in Cartesian coordinates */
    sym_op.Rc = dot(dot(lattice_vectors__, r3::matrix<double>(sym_op.R)), inverse_lattice_vectors);
    /* proper rotation in Cartesian coordinates */
    sym_op.Rcp = sym_op.Rc * p;
    try {
        /* get Euler angles of the rotation */
        sym_op.euler_angles = euler_angles(sym_op.Rcp, tolerance__);
    } catch (std::exception const& e) {
        std::stringstream s;
        s << e.what() << std::endl;
        s << "number of symmetry operations found by spglib: " << spg_dataset__->n_operations << std::endl
          << "symmetry operation: " << isym_spg__ << std::endl
          << "rotation matrix in lattice coordinates: " << sym_op.R << std::endl
          << "rotation matrix in Cartesian coordinates: " << sym_op.Rc << std::endl
          << "lattice vectors: " << lattice_vectors__ << std::endl
          << "metric tensor error: " << metric_tensor_error(lattice_vectors__, sym_op.R) << std::endl
          << std::endl
          << "possible solution: decrease the spglib_tolerance";
        RTE_THROW(s);
    }
    try {
        /* get symmetry related atoms */
        auto result           = find_sym_atom(num_atoms__, positions__, sym_op.R, sym_op.t, tolerance__);
        sym_op.sym_atom       = result.first;
        result                = find_sym_atom(num_atoms__, positions__, sym_op.R, sym_op.t, tolerance__, true);
        sym_op.inv_sym_atom   = result.first;
        sym_op.inv_sym_atom_T = result.second;
        for (int ia = 0; ia < num_atoms__; ia++) {
            int ja = sym_op.sym_atom[ia];
            if (sym_op.inv_sym_atom[ja] != ia) {
                std::stringstream s;
                s << "atom symmetry tables are not consistent" << std::endl
                  << "ia: " << ia << " sym_atom[ia]: " << ja
                  << " inv_sym_atom[sym_atom[ia]]: " << sym_op.inv_sym_atom[ja];
                RTE_THROW(s);
            }
        }
    } catch (std::exception const& e) {
        std::stringstream s;
        s << e.what() << std::endl;
        s << "R: " << sym_op.R << std::endl << "t: " << sym_op.t << std::endl << "tolerance: " << tolerance__;
        RTE_THROW(s);
    }

    return sym_op;
}

static space_group_symmetry_descriptor
get_identity_spg_sym_op(int num_atoms__)
{
    space_group_symmetry_descriptor sym_op;

    sym_op.R = r3::matrix<int>({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    /* inverse of the rotation matrix */
    sym_op.invR = inverse(sym_op.R);
    /* inverse transpose */
    sym_op.invRT = transpose(sym_op.invR);
    /* fractional translation */
    sym_op.t = r3::vector<double>(0, 0, 0);
    /* is this proper or improper rotation */
    sym_op.proper = 1;
    /* rotation in Cartesian coordinates */
    sym_op.Rcp = sym_op.Rc = r3::matrix<double>({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    /* get Euler angles of the rotation */
    sym_op.euler_angles = euler_angles(sym_op.Rc, 1e-10);
    /* trivial atom symmetry table */
    sym_op.sym_atom = std::vector<int>(num_atoms__);
    std::iota(sym_op.sym_atom.begin(), sym_op.sym_atom.end(), 0);
    /* trivial atom symmetry table for inverse operation */
    sym_op.inv_sym_atom = std::vector<int>(num_atoms__);
    std::iota(sym_op.inv_sym_atom.begin(), sym_op.inv_sym_atom.end(), 0);
    sym_op.inv_sym_atom_T = std::vector<r3::vector<int>>(num_atoms__, r3::vector<int>(0, 0, 0));

    return sym_op;
}

Crystal_symmetry::Crystal_symmetry(r3::matrix<double> const& lattice_vectors__, int num_atoms__, int num_atom_types__,
                                   std::vector<int> const& types__, mdarray<double, 2> const& positions__,
                                   mdarray<double, 2> const& spins__, bool spin_orbit__, double tolerance__,
                                   bool use_sym__)
    : lattice_vectors_(lattice_vectors__)
    , num_atoms_(num_atoms__)
    , num_atom_types_(num_atom_types__)
    , types_(types__)
    , tolerance_(tolerance__)
{
    PROFILE("sirius::Crystal_symmetry");

    /* check lattice vectors */
    if (lattice_vectors__.det() < 0 && use_sym__) {
        std::stringstream s;
        s << "spglib requires positive determinant for a matrix of lattice vectors";
        RTE_THROW(s);
    }

    /* make inverse */
    inverse_lattice_vectors_ = inverse(lattice_vectors_);

    double lattice[3][3];
    for (int i : {0, 1, 2}) {
        for (int j : {0, 1, 2}) {
            lattice[i][j] = lattice_vectors_(i, j);
        }
    }

    positions_ = mdarray<double, 2>({3, num_atoms_});
    copy(positions__, positions_);

    magnetization_ = mdarray<double, 2>({3, num_atoms_});
    copy(spins__, magnetization_);

    PROFILE_START("sirius::Crystal_symmetry|spg");
    if (use_sym__) {
        /* make a call to spglib */
        spg_dataset_ = spg_get_dataset(lattice, (double(*)[3]) & positions_(0, 0), &types_[0], num_atoms_, tolerance_);
        if (spg_dataset_ == NULL) {
            RTE_THROW("spg_get_dataset() returned NULL");
        }

        if (spg_dataset_->spacegroup_number == 0) {
            RTE_THROW("spg_get_dataset() returned 0 for the space group");
        }

        if (spg_dataset_->n_atoms != num_atoms__) {
            std::stringstream s;
            s << "spg_get_dataset() returned wrong number of atoms (" << spg_dataset_->n_atoms << ")" << std::endl
              << "expected number of atoms is " << num_atoms__;
            RTE_THROW(s);
        }
    }
    PROFILE_STOP("sirius::Crystal_symmetry|spg");

    if (spg_dataset_) {
        auto lat_sym = find_lat_sym(lattice_vectors_, tolerance_);
        /* make a list of crystal symmetries */
        for (int isym = 0; isym < spg_dataset_->n_operations; isym++) {
            r3::matrix<int> R(spg_dataset_->rotations[isym]);
            for (auto& e : lat_sym) {
                if (e == R) {
                    auto sym_op = get_spg_sym_op(isym, spg_dataset_, lattice_vectors__, num_atoms__, positions__,
                                                 tolerance__);
                    /* add symmetry operation to a list */
                    space_group_symmetry_.push_back(sym_op);
                }
            }
        }
    } else { /* add only identity element */
        auto sym_op = get_identity_spg_sym_op(num_atoms__);
        /* add symmetry operation to a list */
        space_group_symmetry_.push_back(sym_op);
    }

    PROFILE_START("sirius::Crystal_symmetry|mag");
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
            auto Rspin = space_group_symmetry(jsym).Rcp;

            int n{0};
            /* check if all atoms transform under spatial and spin symmetries */
            for (int ia = 0; ia < num_atoms_; ia++) {
                int ja = space_group_symmetry(isym).sym_atom[ia];

                /* now check that vector field transforms from atom ia to atom ja */
                /* vector field of atom is expected to be in Cartesian coordinates */
                auto vd = dot(Rspin, r3::vector<double>(spins__(0, ia), spins__(1, ia), spins__(2, ia))) -
                          r3::vector<double>(spins__(0, ja), spins__(1, ja), spins__(2, ja));

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
    PROFILE_STOP("sirius::Crystal_symmetry|mag");
}

double
Crystal_symmetry::metric_tensor_error() const
{
    double diff{0};
    for (auto const& e : magnetic_group_symmetry_) {
        diff = std::max(diff, sirius::metric_tensor_error(lattice_vectors_, e.spg_op.R));
    }
    return diff;
}

double
Crystal_symmetry::sym_op_R_error() const
{
    double diff{0};
    for (auto const& e : magnetic_group_symmetry_) {
        auto R  = e.spg_op.Rcp;
        auto R1 = inverse(transpose(R));
        for (int i : {0, 1, 2}) {
            for (int j : {0, 1, 2}) {
                diff = std::max(diff, std::abs(R1(i, j) - R(i, j)));
            }
        }
    }
    return diff;
}

void
Crystal_symmetry::print_info(std::ostream& out__, int verbosity__) const
{
    if (this->spg_dataset_ && (this->spg_dataset_->n_operations != this->num_spg_sym())) {
        out__ << "space group found by spglib is different" << std::endl
              << "  num. sym. spglib : " << this->spg_dataset_->n_operations << std::endl
              << "  num. sym. actual : " << this->num_spg_sym() << std::endl
              << "  tolerance : " << this->tolerance_ << std::endl;
    } else {
        out__ << "space group number   : " << this->spacegroup_number() << std::endl
              << "international symbol : " << this->international_symbol() << std::endl
              << "Hall symbol          : " << this->hall_symbol() << std::endl
              << "space group transformation matrix : " << std::endl;
        auto tm = this->transformation_matrix();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                out__ << ffmt(8, 4) << tm(i, j);
            }
            out__ << std::endl;
        }
        out__ << "space group origin shift : " << std::endl;
        auto t = this->origin_shift();
        for (auto x : {0, 1, 2}) {
            out__ << ffmt(8, 4) << t[x];
        }
        out__ << std::endl;
    }
    out__ << "number of space group operations  : " << this->num_spg_sym() << std::endl
          << "number of magnetic group operations : " << this->size() << std::endl
          << "metric tensor error: " << std::scientific << this->metric_tensor_error() << std::endl
          << "rotation matrix error: " << std::scientific << this->sym_op_R_error() << std::endl;

    if (verbosity__ >= 2) {
        out__ << std::endl << "symmetry operations " << std::endl << std::endl;
        for (int isym = 0; isym < this->size(); isym++) {
            auto R  = this->operator[](isym).spg_op.R;
            auto Rc = this->operator[](isym).spg_op.Rc;
            auto t  = this->operator[](isym).spg_op.t;
            auto S  = this->operator[](isym).spin_rotation;

            out__ << "isym : " << isym << std::endl << "R : ";
            for (int i : {0, 1, 2}) {
                if (i) {
                    out__ << "    ";
                }
                for (int j : {
                             0,
                             1,
                             2,
                     }) {
                    out__ << std::setw(3) << R(i, j);
                }
                out__ << std::endl;
            }
            out__ << "Rc: ";
            for (int i : {0, 1, 2}) {
                if (i) {
                    out__ << "    ";
                }
                for (int j : {0, 1, 2}) {
                    out__ << ffmt(8, 4) << Rc(i, j);
                }
                out__ << std::endl;
            }
            out__ << "t : ";
            for (int j : {0, 1, 2}) {
                out__ << ffmt(8, 4) << t[j];
            }
            out__ << std::endl;
            out__ << "S : ";
            for (int i : {0, 1, 2}) {
                if (i) {
                    out__ << "    ";
                }
                for (int j : {0, 1, 2}) {
                    out__ << ffmt(8, 4) << S(i, j);
                }
                out__ << std::endl;
            }
            out__ << "proper: " << std::setw(2) << this->operator[](isym).spg_op.proper << std::endl << std::endl;
        }
    }
}

} // namespace sirius
