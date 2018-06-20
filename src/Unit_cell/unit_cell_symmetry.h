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

/** \file unit_cell_symmetry.h
 *
 *  \brief Contains definition and implementation of sirius::Unit_cell_symmetry class.
 */

#ifndef __UNIT_CELL_SYMMETRY_H__
#define __UNIT_CELL_SYMMETRY_H__

extern "C" {
#ifdef __SPGLIB_STD_INCLUDE_PATH
#include <spglib/spglib.h>
#else
#include <spglib.h>
#endif
}

#include "geometry3d.hpp"
#include "constants.h"
#include "utils.h"
#include "gvec.hpp"

namespace sirius {

/// Descriptor of the space group symmetry operation.
struct space_group_symmetry_descriptor
{
    /// Rotational part of symmetry operation (fractional coordinates).
    matrix3d<int> R;
    /// inverse of R
    matrix3d<int> invR;
    /// inverse transposed of R
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
    /// inverse of proper spin rotation matrix in Cartesian coordinates.
    matrix3d<double> spin_rotation_inv;

};

class Unit_cell_symmetry
{
    private:

        /// Matrix of lattice vectors.
        /** Spglib requires this matrix to have a positively defined determinant. */
        matrix3d<double> lattice_vectors_;

        matrix3d<double> inverse_lattice_vectors_;

        int num_atoms_;

        mdarray<double, 2> positions_;

        std::vector<int> types_;

        double tolerance_;

        /// Crystal structure descriptor returned by spglib.
        SpglibDataset* spg_dataset_;

        /// Symmetry table for atoms.
        /** For each atom ia and symmetry isym sym_table_(ia, isym) stores index of atom ja to which original atom
         *  transforms under symmetry operation. */
        mdarray<int, 2> sym_table_;

        /// List of all space group symmetry operations.
        std::vector<space_group_symmetry_descriptor> space_group_symmetry_;

        /// List of all magnetic group symmetry operations.
        std::vector<magnetic_group_symmetry_descriptor> magnetic_group_symmetry_;

        /// Compute Euler angles corresponding to the proper rotation part of the given symmetry.
        vector3d<double> euler_angles(matrix3d<double> const& rot__) const;

        /// Generate rotation matrix from three Euler angles
        /** Euler angles \f$ \alpha, \beta, \gamma \f$ define the general rotation as three consecutive rotations:
         *      - about \f$ \hat e_z \f$ through the angle \f$ \gamma \f$ (\f$ 0 \le \gamma < 2\pi \f$)
         *      - about \f$ \hat e_y \f$ through the angle \f$ \beta \f$ (\f$ 0 \le \beta \le \pi \f$)
         *      - about \f$ \hat e_z \f$ through the angle \f$ \alpha \f$ (\f$ 0 \le \gamma < 2\pi \f$)
         *
         *  The total rotation matrix is defined as a product of three rotation matrices:
         *  \f[
         *      R(\alpha, \beta, \gamma) =
         *          \left( \begin{array}{ccc} \cos(\alpha) & -\sin(\alpha) & 0 \\
         *                                    \sin(\alpha) & \cos(\alpha) & 0 \\
         *                                    0 & 0 & 1 \end{array} \right)
         *          \left( \begin{array}{ccc} \cos(\beta) & 0 & \sin(\beta) \\
         *                                    0 & 1 & 0 \\
         *                                    -\sin(\beta) & 0 & \cos(\beta) \end{array} \right)
         *          \left( \begin{array}{ccc} \cos(\gamma) & -\sin(\gamma) & 0 \\
         *                                    \sin(\gamma) & \cos(\gamma) & 0 \\
         *                                    0 & 0 & 1 \end{array} \right) =
         *      \left( \begin{array}{ccc} \cos(\alpha) \cos(\beta) \cos(\gamma) - \sin(\alpha) \sin(\gamma) &
         *                                -\sin(\alpha) \cos(\gamma) - \cos(\alpha) \cos(\beta) \sin(\gamma) &
         *                                \cos(\alpha) \sin(\beta) \\
         *                                \sin(\alpha) \cos(\beta) \cos(\gamma) + \cos(\alpha) \sin(\gamma) &
         *                                \cos(\alpha) \cos(\gamma) - \sin(\alpha) \cos(\beta) \sin(\gamma) &
         *                                \sin(\alpha) \sin(\beta) \\
         *                                -\sin(\beta) \cos(\gamma) &
         *                                \sin(\beta) \sin(\gamma) &
         *                                \cos(\beta) \end{array} \right)
         *  \f]
         */
        matrix3d<double> rot_mtrx_cart(vector3d<double> euler_angles__) const;

        /// Get axis and angle from rotation matrix.
        static std::pair<vector3d<double>, double> axis_angle(matrix3d<double> R__)
        {
            vector3d<double> u;
            /* make proper rotation */
            R__ = R__ * R__.det();
            u[0] = R__(2, 1) - R__(1, 2);
            u[1] = R__(0, 2) - R__(2, 0);
            u[2] = R__(1, 0) - R__(0, 1);

            double sint = u.length() / 2.0;
            double cost = (R__(0, 0) + R__(1, 1) + R__(2, 2) - 1) / 2.0;

            double theta = utils::phi_by_sin_cos(sint, cost);

            /* rotation angle is zero */
            if (std::abs(theta) < 1e-12) {
                u = vector3d<double>({0, 0, 1});
            } else if (std::abs(theta - pi) < 1e-12) { /* rotation angle is Pi */
                /* rotation matrix for Pi angle has this form

                [-1+2ux^2 |  2 ux uy |  2 ux uz]
                [2 ux uy  | -1+2uy^2 |  2 uy uz]
                [2 ux uz  | 2 uy uz  | -1+2uz^2] */

                if (R__(0, 0) >= R__(1, 1) && R__(0, 0) >= R__(2, 2)) { /* x-component is largest */
                    u[0] = std::sqrt(std::abs(R__(0, 0) + 1) / 2);
                    u[1] = (R__(0, 1) + R__(1, 0)) / 4 / u[0];
                    u[2] = (R__(0, 2) + R__(2, 0)) / 4 / u[0];
                } else if (R__(1, 1) >= R__(0, 0) && R__(1, 1) >= R__(2, 2)) { /* y-component is largest */
                    u[1] = std::sqrt(std::abs(R__(1, 1) + 1) / 2);
                    u[0] = (R__(1, 0) + R__(0, 1)) / 4 / u[1];
                    u[2] = (R__(1, 2) + R__(2, 1)) / 4 / u[1];
                } else {
                    u[2] = std::sqrt(std::abs(R__(2, 2) + 1) / 2);
                    u[0] = (R__(2, 0) + R__(0, 2)) / 4 / u[2];
                    u[1] = (R__(2, 1) + R__(1, 2)) / 4 / u[2];
                }
            } else {
                u = u * (1.0 / u.length());
            }

            return std::pair<vector3d<double>, double>(u, theta);
        }

        static mdarray<double_complex, 2> spinor_rotation_matrix(vector3d<double> u__, double theta__)
        {
            mdarray<double_complex, 2> rotm(2, 2);

            auto cost = std::cos(theta__ / 2);
            auto sint = std::sin(theta__ / 2);

            rotm(0, 0) = double_complex(cost, -u__[2] * sint);
            rotm(1, 1) = double_complex(cost,  u__[2] * sint);
            rotm(0, 1) = double_complex(-u__[1] * sint, -u__[0] * sint);
            rotm(1, 0) = double_complex( u__[1] * sint, -u__[0] * sint);

            return std::move(rotm);
        }


    public:

        Unit_cell_symmetry(matrix3d<double>&   lattice_vectors__,
                           int                 num_atoms__,
                           mdarray<double, 2>& positions__,
                           mdarray<double, 2>& spins__,
                           std::vector<int>&   types__,
                           double              tolerance__);

        ~Unit_cell_symmetry()
        {
            spg_free_dataset(spg_dataset_);
        }

        inline int atom_symmetry_class(int ia__)
        {
            return spg_dataset_->equivalent_atoms[ia__];
        }

        inline int spacegroup_number()
        {
            return spg_dataset_->spacegroup_number;
        }

        inline std::string international_symbol()
        {
            return spg_dataset_->international_symbol;
        }

        inline std::string hall_symbol()
        {
            return spg_dataset_->hall_symbol;
        }

        matrix3d<double> transformation_matrix() const
        {
           return matrix3d<double>(spg_dataset_->transformation_matrix);
        }

        vector3d<double> origin_shift() const
        {
            return vector3d<double>(spg_dataset_->origin_shift[0],
                                    spg_dataset_->origin_shift[1],
                                    spg_dataset_->origin_shift[2]);
        }

        inline int num_spg_sym() const
        {
            return static_cast<int>(space_group_symmetry_.size());
        }

        inline space_group_symmetry_descriptor const& space_group_symmetry(int isym__) const
        {
            assert(isym__ >= 0 && isym__ < num_spg_sym());
            return space_group_symmetry_[isym__];
        }
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

        void check_gvec_symmetry(Gvec const& gvec__, Communicator const& comm__) const;

        /// Symmetrize scalar function.
        /** Symmetrize scalar function. The following operation is performed:
         *   \f[
         *     f_{\mathrm{sym}}({\bf x}) = \frac{1}{N_{\mathrm{sym}}} \sum_{{\bf \hat P}} f({\bf \hat P x})
         *   \f]
         *   For the function expanded in plane-waves we have:
         *   \f[
         *     f_{\mathrm{sym}}({\bf x}) = \frac{1}{N_{\mathrm{sym}}} \sum_{{\bf \hat P}} \sum_{\bf G}
         *     e^{i{\bf G \hat P x}} \hat f({\bf G})
         *                = \frac{1}{N_{\mathrm{sym}}} \sum_{{\bf \hat P}} \sum_{\bf G} e^{i{\bf G (Rx +
         *                    t)}} \hat f({\bf G})
         *                = \frac{1}{N_{\mathrm{sym}}} \sum_{{\bf \hat P}} \sum_{\bf G} e^{i{\bf G t}}
         *                e^{i{\bf R^T G x}} \hat f({\bf G})
         *   \f]
         *   Substitute \f$\bf \tilde G = \bf R^T \bf G\f$
         *   \f[
         *     f_{\mathrm{sym}}({\bf x}) =
         *     \frac{1}{N_{\mathrm{sym}}} \sum_{{\bf \tilde G}} e^{i {\bf \tilde G} }\sum_{{\bf \hat P}}
         *     e^{i {\bf R}^{-T} {\bf t}} \hat f ({\bf R}^{-T} {\bf \tilde G}) \,,
         *   \f]
         *   to find the Fourier coefficients \f$ \hat f_{\mathrm{sym}} \f$ of \f$f_{\mathrm{sym}}\f$ in
         *   terms of \f$ \hat f \f$:
         *   \f[
         *     \hat f_{\mathrm{sym}}({\bf G}) = \frac{1}{N_{\mathrm{sym}}} \sum_{{\bf \hat P}} e^{i {\bf R}^{-T} {\bf t}} \hat f ({\bf R}^{-T} {\bf G})\,.
         *   \f]
         *   Once \f$\hat f_{\mathrm{sym}} \f$ has been calculated by the above formula for a single \f$\bf G \f$, its values at
         *   points \f${\bf R}^{-T} {\bf G}\f$, \f$\forall \, {\bf R}\f$ are given by the update formula
         *   \f[
         *     \hat f_{\mathrm{sym}} ({\bf R}^{-T} {\bf G}) = e^{-i {\bf R}^{-T} {\bf G} {\bf t}}
         *     \hat{f}_{\mathrm{sym}} ({\bf G})\,,
         *   \f]
         *
         *   which follows by using that \f$f_{\mathrm{sym}}({\bf \hat P} {\bf x}) = f_{\mathrm{sym}}({\bf x})\f$:
         *   \f{eqnarray*}{
         *     f_{\mathrm{sym}}(\hat{P}{\bf x}) &=& \sum_G e^{i G ({\bf R}{\bf x} + {\bf t} )} \hat{f}_{\mathrm{sym}}({\bf G}) \ \
                                                &=& \sum_G e^{i {\bf G} {\bf x}} e^{i {\bf R}^{-T} {\bf G} {\bf t}} \hat{f}_{\mathrm{sym}}({\bf R}^{-T} {\bf G}) \,.
         *   \f}
         */
         void symmetrize_function(double_complex* f_pw__,
                                 remap_gvec_to_shells const& remap_gvec__,
                                 mdarray<double_complex, 3> const& sym_phase_factors__) const;


         void symmetrize_vector_function(double_complex* fz_pw__,
                                         remap_gvec_to_shells const& remap_gvec__,
                                         mdarray<double_complex, 3> const& sym_phase_factors__) const;

         /**
          *   Symmetrize vector valued function.
          *
          *   The following operations are performed.
          *
          *   Fourier coefficient of symmetrized function:
          *   \f[
          *     \hat f_{\mathrm{sym}}({\bf G}) = \frac{1}{N_{\mathrm{sym}}} \sum_{{\bf \hat P}} e^{i {\bf R}^{-T} {\bf t}} {\bf S} \hat f ({\bf R}^{-T} {\bf G})\,.
          *   \f]
          *
          *   Update formula when \f$\hat f({\bf G})\f$ is known:
          *   \f[
          *     \hat f_{\mathrm{sym}} ({\bf R}^{-T} {\bf G}) = e^{-i {\bf R}^{-T} {\bf G} {\bf t}}
          *     {\bf S}^{-1} \hat{f}_{\mathrm{sym}} ({\bf G})\,,
          *   \f]
          *
          *   The derivation works analogically to the one for Unit_cell_symmetry#symmetrize_function .
          *
          */
          void symmetrize_vector_function(double_complex* fx_pw__,
                                          double_complex* fy_pw__,
                                          double_complex* fz_pw__,
                                          remap_gvec_to_shells const& remap_gvec__,
                                          mdarray<double_complex, 3> const& sym_phase_factors__) const;

          void symmetrize_function(mdarray<double, 3>& frlm__,
                                   Communicator const& comm__) const;

          void symmetrize_vector_function(mdarray<double, 3>& fz_rlm__,
                                          Communicator const& comm__) const;

          void symmetrize_vector_function(mdarray<double, 3>& fx_rlm__,
                                          mdarray<double, 3>& fy_rlm__,
                                          mdarray<double, 3>& fz_rlm__,
                                          Communicator const& comm__) const;

          int get_irreducible_reciprocal_mesh(vector3d<int> k_mesh__,
                                              vector3d<int> is_shift__,
                                              mdarray<double, 2>& kp__,
                                              std::vector<double>& wk__) const;

          matrix3d<double> const& lattice_vectors() const
          {
              return lattice_vectors_;
          }

          matrix3d<double> const& inverse_lattice_vectors() const
          {
              return inverse_lattice_vectors_;
          }
};

inline Unit_cell_symmetry::Unit_cell_symmetry(matrix3d<double>&   lattice_vectors__,
                                              int                 num_atoms__,
                                              mdarray<double, 2>& positions__,
                                              mdarray<double, 2>& spins__,
                                              std::vector<int>&   types__,
                                              double              tolerance__)
    : lattice_vectors_(lattice_vectors__)
    , num_atoms_(num_atoms__)
    , types_(types__)
    , tolerance_(tolerance__)
{
    PROFILE("sirius::Unit_cell_symmetry::Unit_cell_symmetry");

    if (lattice_vectors__.det() < 0) {
        std::stringstream s;
        s << "spglib requires positive determinant for a matrix of lattice vectors";
        TERMINATE(s);
    }

    double lattice[3][3];
    for (int i: {0, 1, 2}) {
        for (int j: {0, 1, 2}) {
            lattice[i][j] = lattice_vectors_(i, j);
        }
    }
    positions_ = mdarray<double, 2>(3, num_atoms_);
    for (int ia = 0; ia < num_atoms_; ia++) {
        for (int x: {0, 1, 2}) {
            positions_(x, ia) = positions__(x, ia);
        }
    }

    utils::timer t1("sirius::Unit_cell_symmetry::Unit_cell_symmetry|spg");
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
    t1.stop();

    inverse_lattice_vectors_ = inverse(lattice_vectors_);

    utils::timer t2("sirius::Unit_cell_symmetry::Unit_cell_symmetry|sym1");
    for (int isym = 0; isym < spg_dataset_->n_operations; isym++) {
        space_group_symmetry_descriptor sym_op;

        sym_op.R = matrix3d<int>(spg_dataset_->rotations[isym]);
        sym_op.invRT = transpose(inverse(sym_op.R));
        sym_op.t = vector3d<double>(spg_dataset_->translations[isym][0],
                                    spg_dataset_->translations[isym][1],
                                    spg_dataset_->translations[isym][2]);
        int p = sym_op.R.det();
        if (!(p == 1 || p == -1)) {
            TERMINATE("wrong rotation matrix");
        }
        sym_op.proper = p;
        sym_op.rotation = lattice_vectors_ * matrix3d<double>(sym_op.R * p) * inverse_lattice_vectors_;
        sym_op.euler_angles = euler_angles(sym_op.rotation);

        for (int i = 0; i < spg_dataset_->n_operations; i++) {
            auto m =  matrix3d<int>(spg_dataset_->rotations[isym]) * matrix3d<int>(spg_dataset_->rotations[i]);

            if (m(0, 0) == 1 && m(1, 1) == 1 && m(2, 2) == 1 &&
                m(0, 1) == 0 && m(1, 0) == 0 &&
                m(0, 2) == 0 && m(2, 0) == 0 &&
                m(1, 2) == 0 && m(2, 1) == 0)
            {
                sym_op.invR = matrix3d<int>(spg_dataset_->rotations[i]);
                break;
            }
        }

        space_group_symmetry_.push_back(sym_op);
    }
    t2.stop();

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
            auto v = reduce_coordinates(R * pos + t);

            int ja = -1;
            /* check for equivalent atom */
            for (int k = 0; k < num_atoms_; k++) {
                vector3d<double> pos1(positions__(0, k), positions__(1, k), positions__(2, k));
                if ((v.first - pos1).length() < tolerance_) {
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
        /* loop over spin symmetries */
        for (int jsym = 0; jsym < num_spg_sym(); jsym++) {
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

inline matrix3d<double> Unit_cell_symmetry::rot_mtrx_cart(vector3d<double> euler_angles) const
{
    double alpha = euler_angles[0];
    double beta = euler_angles[1];
    double gamma = euler_angles[2];

    matrix3d<double> rm;
    rm(0, 0) = std::cos(alpha) * std::cos(beta) * std::cos(gamma) - std::sin(alpha) * std::sin(gamma);
    rm(0, 1) = -std::cos(gamma) * std::sin(alpha) - std::cos(alpha) * std::cos(beta) * std::sin(gamma);
    rm(0, 2) = std::cos(alpha) * std::sin(beta);
    rm(1, 0) = std::cos(beta) * std::cos(gamma) * std::sin(alpha) + std::cos(alpha) * std::sin(gamma);
    rm(1, 1) = std::cos(alpha) * std::cos(gamma) - std::cos(beta) * std::sin(alpha) * std::sin(gamma);
    rm(1, 2) = std::sin(alpha) * std::sin(beta);
    rm(2, 0) = -std::cos(gamma) * std::sin(beta);
    rm(2, 1) = std::sin(beta) * std::sin(gamma);
    rm(2, 2) = std::cos(beta);

    return rm;
}

inline vector3d<double> Unit_cell_symmetry::euler_angles(matrix3d<double> const& rot__) const
{
    vector3d<double> angles(0, 0, 0);

    if (std::abs(rot__.det() - 1) > 1e-10) {
        std::stringstream s;
        s << "determinant of rotation matrix is " << rot__.det();
        TERMINATE(s);
    }

    if (std::abs(rot__(2, 2) - 1.0) < 1e-10) { // cos(beta) == 1, beta = 0
        angles[0] = utils::phi_by_sin_cos(rot__(1, 0), rot__(0, 0));
    } else if (std::abs(rot__(2, 2) + 1.0) < 1e-10) { // cos(beta) == -1, beta = Pi
        angles[0] = utils::phi_by_sin_cos(-rot__(0, 1), rot__(1, 1));
        angles[1] = pi;
    } else {
        double beta = std::acos(rot__(2, 2));
        angles[0] = utils::phi_by_sin_cos(rot__(1, 2) / std::sin(beta), rot__(0, 2) / std::sin(beta));
        angles[1] = beta;
        angles[2] = utils::phi_by_sin_cos(rot__(2, 1) / std::sin(beta), -rot__(2, 0) / std::sin(beta));
    }

    auto rm1 = rot_mtrx_cart(angles);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (std::abs(rot__(i, j) - rm1(i, j)) > 1e-8) {
                std::stringstream s;
                s << "matrices don't match" << std::endl
                  << "initial symmetry matrix: " << std::endl
                  << rot__(0, 0) << " " << rot__(0, 1) << " " << rot__(0, 2) << std::endl
                  << rot__(1, 0) << " " << rot__(1, 1) << " " << rot__(1, 2) << std::endl
                  << rot__(2, 0) << " " << rot__(2, 1) << " " << rot__(2, 2) << std::endl
                  << "euler angles : " << angles[0] / pi << " " << angles[1] / pi << " " << angles[2] / pi << std::endl
                  << "computed symmetry matrix : " << std::endl
                  << rm1(0, 0) << " " << rm1(0, 1) << " " << rm1(0, 2) << std::endl
                  << rm1(1, 0) << " " << rm1(1, 1) << " " << rm1(1, 2) << std::endl
                  << rm1(2, 0) << " " << rm1(2, 1) << " " << rm1(2, 2) << std::endl;
                TERMINATE(s);
            }
        }
    }

    return angles;
}

inline int Unit_cell_symmetry::get_irreducible_reciprocal_mesh(vector3d<int> k_mesh__,
                                                               vector3d<int> is_shift__,
                                                               mdarray<double, 2>& kp__,
                                                               std::vector<double>& wk__) const
{
    int nktot = k_mesh__[0] * k_mesh__[1] * k_mesh__[2];

    mdarray<int, 2> grid_address(3, nktot);
    std::vector<int> ikmap(nktot);

    double lattice[3][3];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++) lattice[i][j] = lattice_vectors_(i, j);
    }

    int nknr = spg_get_ir_reciprocal_mesh((int(*)[3])&grid_address(0, 0),
                                          &ikmap[0],
                                          &k_mesh__[0],
                                          &is_shift__[0],
                                          1,
                                          lattice,
                                          (double(*)[3])&positions_(0, 0),
                                          &types_[0],
                                          num_atoms_,
                                          tolerance_);

    std::map<int, int> wknr;
    for (int ik = 0; ik < nktot; ik++)
    {
        if (wknr.count(ikmap[ik]) == 0) wknr[ikmap[ik]] = 0;
        wknr[ikmap[ik]] += 1;
    }

    wk__ = std::vector<double>(nknr);
    kp__ = mdarray<double, 2>(3, nknr);

    int n = 0;
    for (auto it = wknr.begin(); it != wknr.end(); it++) {
        wk__[n] = double(it->second) / nktot;
        for (int x = 0; x < 3; x++) {
            kp__(x, n) = double(grid_address(x, it->first) + is_shift__[x] / 2.0) / k_mesh__[x];
        }
        n++;
    }

    return nknr;
}

inline void Unit_cell_symmetry::check_gvec_symmetry(Gvec const& gvec__, Communicator const& comm__) const
{
    PROFILE("sirius::Unit_cell_symmetry::check_gvec_symmetry");

    int gvec_count  = gvec__.gvec_count(comm__.rank());
    int gvec_offset = gvec__.gvec_offset(comm__.rank());

    #pragma omp parallel for
    for (int isym = 0; isym < num_mag_sym(); isym++) {
        auto sm = magnetic_group_symmetry(isym).spg_op.R;

        for (int igloc = 0; igloc < gvec_count; igloc++) {
            int ig = gvec_offset + igloc;

            auto gv = gvec__.gvec(ig);
            /* apply symmetry operation to the G-vector */
            auto gv_rot = transpose(sm) * gv;

            //== /* check limits */
            //== for (int x: {0, 1, 2}) {
            //==     auto limits = gvec__.fft_box().limits(x);
            //==     /* check boundaries */
            //==     if (gv_rot[x] < limits.first || gv_rot[x] > limits.second) {
            //==         std::stringstream s;
            //==         s << "rotated G-vector is outside of grid limits" << std::endl
            //==           << "original G-vector: " << gv << ", length: " << gvec__.cart(ig).length() << std::endl
            //==           << "rotation matrix: " << std::endl
            //==           << sm(0, 0) << " " << sm(0, 1) << " " << sm(0, 2) << std::endl
            //==           << sm(1, 0) << " " << sm(1, 1) << " " << sm(1, 2) << std::endl
            //==           << sm(2, 0) << " " << sm(2, 1) << " " << sm(2, 2) << std::endl
            //==           << "rotated G-vector: " << gv_rot << std::endl
            //==           << "limits: "
            //==           << gvec__.fft_box().limits(0).first << " " <<  gvec__.fft_box().limits(0).second << " "
            //==           << gvec__.fft_box().limits(1).first << " " <<  gvec__.fft_box().limits(1).second << " "
            //==           << gvec__.fft_box().limits(2).first << " " <<  gvec__.fft_box().limits(2).second;

            //==           TERMINATE(s);
            //==     }
            //== }
            int ig_rot = gvec__.index_by_gvec(gv_rot);
            /* special case where -G is equal to G */
            if (gvec__.reduced() && ig_rot < 0) {
                gv_rot = gv_rot * (-1);
                ig_rot = gvec__.index_by_gvec(gv_rot);
            }
            if (ig_rot < 0 || ig_rot >= gvec__.num_gvec()) {
                std::stringstream s;
                s << "rotated G-vector index is wrong" << std::endl
                  << "original G-vector: " << gv << std::endl
                  << "rotation matrix: " << std::endl
                  << sm(0, 0) << " " << sm(0, 1) << " " << sm(0, 2) << std::endl
                  << sm(1, 0) << " " << sm(1, 1) << " " << sm(1, 2) << std::endl
                  << sm(2, 0) << " " << sm(2, 1) << " " << sm(2, 2) << std::endl
                  << "rotated G-vector: " << gv_rot << std::endl
                  << "rotated G-vector index: " << ig_rot << std::endl
                  << "number of G-vectors: " << gvec__.num_gvec();
                  TERMINATE(s);
            }
        }
    }
}


inline void Unit_cell_symmetry::symmetrize_function(double_complex* f_pw__,
                                         remap_gvec_to_shells const& remap_gvec__,
                                         mdarray<double_complex, 3> const& sym_phase_factors__) const
{
   PROFILE("sirius::Unit_cell_symmetry::symmetrize_function_pw");

   auto v = remap_gvec__.remap_forward(f_pw__);

   std::vector<double_complex> sym_f_pw(v.size(), 0);
   std::vector<bool> is_done(v.size(), false);

   double norm = 1 / double(num_mag_sym());

   auto phase_factor = [&](int isym, vector3d<int> G)
   {
       return sym_phase_factors__(0, G[0], isym) *
              sym_phase_factors__(1, G[1], isym) *
              sym_phase_factors__(2, G[2], isym);
   };

   utils::timer t1("sirius::Unit_cell_symmetry::symmetrize_function_pw|local");

   #pragma omp parallel
   {
       int nt = omp_get_max_threads();
       int tid = omp_get_thread_num();

       for (int igloc = 0; igloc < remap_gvec__.a2a_recv.size(); igloc++) {
           vector3d<int> G(&remap_gvec__.gvec_remapped_(0, igloc));

           int igsh = remap_gvec__.gvec_shell_remapped(igloc);

           /* each thread is working on full shell of G-vectors */
           if (igsh % nt == tid && !is_done[igloc]) {
               double_complex zsym(0, 0);

               for (int i = 0; i < num_mag_sym(); i++) {
                   const auto& invRT = magnetic_group_symmetry(i).spg_op.invRT;
                   auto gv_rot = invRT * G;
                   double_complex phase = phase_factor(i, gv_rot);

                   /* index of a rotated G-vector */
                   int ig_rot = remap_gvec__.index_by_gvec(gv_rot);

                   if (ig_rot == -1) {
                       gv_rot = gv_rot * (-1);
                       ig_rot = remap_gvec__.index_by_gvec(gv_rot);
                       assert(ig_rot >= 0 && ig_rot < (int)v.size());
                       zsym += std::conj(v[ig_rot]) * phase;
                   } else {
                       assert(ig_rot >= 0 && ig_rot < (int)v.size());
                       zsym += v[ig_rot] * phase;
                   }
               } /* loop over symmetries */

               zsym *= norm;

               for (int i = 0; i < num_mag_sym(); i++) {
                   const auto& invRT = magnetic_group_symmetry(i).spg_op.invRT;
                   auto gv_rot = invRT * G;
                   /* index of a rotated G-vector */
                   int ig_rot = remap_gvec__.index_by_gvec(gv_rot);
                   double_complex phase = std::conj(phase_factor(i, gv_rot));

                   if (ig_rot == -1) {
                       /* skip */
                   } else {
                       assert(ig_rot >= 0 && ig_rot < int(v.size()));
                       sym_f_pw[ig_rot] = zsym * phase;
                       is_done[ig_rot] = true;
                   }
               } /* loop over symmetries */
           }
       } /* loop over igloc */
   }
   t1.stop();

   remap_gvec__.remap_backward(sym_f_pw, f_pw__);
}

inline void Unit_cell_symmetry::symmetrize_vector_function(double_complex* fz_pw__,
                                                           remap_gvec_to_shells const& remap_gvec__,
                                                           mdarray<double_complex, 3> const& sym_phase_factors__) const
{
    PROFILE("sirius::Unit_cell_symmetry::symmetrize_vector_function_pw_1c");

    auto phase_factor = [&](int isym, const vector3d<int>& G) {
        return sym_phase_factors__(0, G[0], isym) *
               sym_phase_factors__(1, G[1], isym) *
               sym_phase_factors__(2, G[2], isym);
    };

    auto v = remap_gvec__.remap_forward(fz_pw__);

    std::vector<double_complex> sym_f_pw(v.size(), 0);
    std::vector<bool> is_done(v.size(), false);
    double norm = 1 / double(num_mag_sym());

    #pragma omp parallel
    {
        int nt = omp_get_max_threads();
        int tid = omp_get_thread_num();

        for (int igloc = 0; igloc < remap_gvec__.a2a_recv.size(); igloc++) {
            vector3d<int> G(&remap_gvec__.gvec_remapped_(0, igloc));

            int igsh = remap_gvec__.gvec_shell_remapped(igloc);

            if (igsh % nt == tid && !is_done[igloc]) {
                double_complex zsym(0, 0);

                for (int i = 0; i < num_mag_sym(); i++) {
                    const auto& invRT = magnetic_group_symmetry(i).spg_op.invRT;
                    const auto& S = magnetic_group_symmetry(i).spin_rotation;
                    double_complex phase = phase_factor(i, G) * S(2, 2);
                    auto gv_rot = invRT * G;
                    /* index of a rotated G-vector */
                    int ig_rot = remap_gvec__.index_by_gvec(gv_rot);

                    if (ig_rot == -1) {
                        gv_rot = gv_rot * (-1);
                        ig_rot = remap_gvec__.index_by_gvec(gv_rot);
                        assert(ig_rot >= 0 && ig_rot < (int)v.size());
                        zsym += std::conj(v[ig_rot]) * phase;
                    } else {
                        assert(ig_rot >= 0 && ig_rot < (int)v.size());
                        zsym += v[ig_rot] * phase;
                    }
                } /* loop over symmetries */

                zsym *= norm;

                for (int i = 0; i < num_mag_sym(); i++) {
                    const auto& invRT = magnetic_group_symmetry(i).spg_op.invRT;
                    const auto& S = magnetic_group_symmetry(i).spin_rotation;
                    auto gv_rot = invRT * G;
                    /* index of rotated G-vector */
                    int ig_rot = remap_gvec__.index_by_gvec(gv_rot);
                    double_complex phase = std::conj(phase_factor(i, gv_rot)) / S(2, 2) ;

                    if (ig_rot == -1) {
                        /* skip */
                    } else {
                        assert(ig_rot >= 0 && ig_rot < int(v.size()));
                        sym_f_pw[ig_rot] = zsym * phase;
                        is_done[ig_rot] = true;
                    }
                } /* loop over symmetries */
            }
        }
    }

    remap_gvec__.remap_backward(sym_f_pw, fz_pw__);
}

inline void Unit_cell_symmetry::symmetrize_vector_function(double_complex* fx_pw__,
                                                           double_complex* fy_pw__,
                                                           double_complex* fz_pw__,
                                                           remap_gvec_to_shells const& remap_gvec__,
                                                           mdarray<double_complex, 3> const& sym_phase_factors__) const
{
    PROFILE("sirius::Unit_cell_symmetry::symmetrize_vector_function_pw_3c");

    auto vx = remap_gvec__.remap_forward(fx_pw__);
    auto vy = remap_gvec__.remap_forward(fy_pw__);
    auto vz = remap_gvec__.remap_forward(fz_pw__);

    std::vector<double_complex> sym_fx_pw(vx.size(), 0);
    std::vector<double_complex> sym_fy_pw(vx.size(), 0);
    std::vector<double_complex> sym_fz_pw(vx.size(), 0);
    std::vector<bool> is_done(vx.size(), false);
    double norm = 1 / double(num_mag_sym());

    auto phase_factor = [&](int isym, const vector3d<int>& G) {
        return sym_phase_factors__(0, G[0], isym) *
               sym_phase_factors__(0, G[1], isym) *
               sym_phase_factors__(0, G[2], isym);
    };

    auto vrot = [&](const vector3d<double_complex>& v, const matrix3d<double>& S) -> vector3d<double_complex> {
        return S * v;
    };


    #pragma omp parallel
    {
        int nt = omp_get_max_threads();
        int tid = omp_get_thread_num();

        for (int igloc = 0; igloc < remap_gvec__.a2a_recv.size(); igloc++) {
            vector3d<int> G(&remap_gvec__.gvec_remapped_(0, igloc));

            int igsh = remap_gvec__.gvec_shell_remapped(igloc);

            if (igsh % nt == tid && !is_done[igloc]) {
                double_complex xsym(0, 0);
                double_complex ysym(0, 0);
                double_complex zsym(0, 0);

                for (int i = 0; i < num_mag_sym(); i++) {
                    /* full space-group symmetry operation is {R|t} */
                    const auto& invRT = magnetic_group_symmetry(i).spg_op.invRT;
                    const auto& S = magnetic_group_symmetry(i).spin_rotation;
                    double_complex phase = phase_factor(i, G);
                    auto gv_rot = invRT * G;
                    /* index of a rotated G-vector */
                    int ig_rot = remap_gvec__.index_by_gvec(gv_rot);

                    if (ig_rot == -1) {
                        gv_rot = gv_rot * (-1);
                        ig_rot = remap_gvec__.index_by_gvec(gv_rot);
                        vector3d<double_complex> v_rot = vrot({vx[ig_rot], vy[ig_rot], vz[ig_rot]}, S);
                        assert(ig_rot >=0 && ig_rot < (int)vx.size());
                        xsym += std::conj(v_rot[0]) * phase;
                        ysym += std::conj(v_rot[1]) * phase;
                        zsym += std::conj(v_rot[2]) * phase;
                    } else {
                        assert(ig_rot >=0 && ig_rot < (int)vx.size());
                        vector3d<double_complex> v_rot = vrot({vx[ig_rot], vy[ig_rot], vz[ig_rot]}, S);
                        xsym += v_rot[0] * phase;
                        ysym += v_rot[1] * phase;
                        zsym += v_rot[2] * phase;
                    }
                } /* loop over symmetries */

                xsym *= norm;
                ysym *= norm;
                zsym *= norm;

                for (int i = 0; i < num_mag_sym(); i++) {
                    const auto& invRT = magnetic_group_symmetry(i).spg_op.invRT;
                    const auto& invS = magnetic_group_symmetry(i).spin_rotation_inv;
                    auto gv_rot = invRT * G;
                    /* index of a rotated G-vector */
                    int ig_rot = remap_gvec__.index_by_gvec(gv_rot);
                    auto v_rot = vrot({xsym, ysym, zsym}, invS);
                    double_complex phase = std::conj(phase_factor(i, gv_rot));

                    if (ig_rot == -1) {
                        /* skip */
                    } else {
                        assert(ig_rot >= 0 && ig_rot < int(v.size()));
                        sym_fx_pw[ig_rot] = v_rot[0] * phase;
                        sym_fy_pw[ig_rot] = v_rot[1] * phase;
                        sym_fz_pw[ig_rot] = v_rot[2] * phase;
                        is_done[ig_rot] = true;
                    }
                } /* loop over symmetries */

            }
        }
    }

    remap_gvec__.remap_backward(sym_fx_pw, fx_pw__);
    remap_gvec__.remap_backward(sym_fy_pw, fy_pw__);
    remap_gvec__.remap_backward(sym_fz_pw, fz_pw__);
}


inline void Unit_cell_symmetry::symmetrize_function(mdarray<double, 3>& frlm__,
                                          Communicator const& comm__) const
{
    PROFILE("sirius::Unit_cell_symmetry::symmetrize_function_mt");

    int lmmax = (int)frlm__.size(0);
    int nrmax = (int)frlm__.size(1);
    if (num_atoms_ != (int)frlm__.size(2)) TERMINATE("wrong number of atoms");

    splindex<block> spl_atoms(num_atoms_, comm__.size(), comm__.rank());

    int lmax = Utils::lmax_by_lmmax(lmmax);

    mdarray<double, 2> rotm(lmmax, lmmax);

    mdarray<double, 3> fsym(lmmax, nrmax, spl_atoms.local_size());
    fsym.zero();

    double alpha = 1.0 / double(num_mag_sym());

    for (int i = 0; i < num_mag_sym(); i++) {
        /* full space-group symmetry operation is {R|t} */
        int pr = magnetic_group_symmetry(i).spg_op.proper;
        auto eang = magnetic_group_symmetry(i).spg_op.euler_angles;
        int isym = magnetic_group_symmetry(i).isym;
        SHT::rotation_matrix(lmax, eang, pr, rotm);

        for (int ia = 0; ia < num_atoms_; ia++) {
            int ja = sym_table_(ia, isym);
            auto location = spl_atoms.location(ja);
            if (location.rank == comm__.rank()) {
                linalg<CPU>::gemm(0, 0, lmmax, nrmax, lmmax, alpha, rotm.at<CPU>(), rotm.ld(),
                                  frlm__.at<CPU>(0, 0, ia), frlm__.ld(), 1.0,
                                  fsym.at<CPU>(0, 0, location.local_index), fsym.ld());
            }
        }
    }
    double* sbuf = spl_atoms.local_size() ? fsym.at<CPU>() : nullptr;
    comm__.allgather(sbuf, frlm__.at<CPU>(),
                     lmmax * nrmax * spl_atoms.global_offset(),
                     lmmax * nrmax * spl_atoms.local_size());
}

inline void Unit_cell_symmetry::symmetrize_vector_function(mdarray<double, 3>& vz_rlm__,
                                                 Communicator const& comm__) const
{
    PROFILE("sirius::Unit_cell_symmetry::symmetrize_vector_function_mt");

    int lmmax = (int)vz_rlm__.size(0);
    int nrmax = (int)vz_rlm__.size(1);

    splindex<block> spl_atoms(num_atoms_, comm__.size(), comm__.rank());

    if (num_atoms_ != (int)vz_rlm__.size(2)) {
        TERMINATE("wrong number of atoms");
    }

    int lmax = Utils::lmax_by_lmmax(lmmax);

    mdarray<double, 2> rotm(lmmax, lmmax);

    mdarray<double, 3> fsym(lmmax, nrmax, spl_atoms.local_size());
    fsym.zero();

    double alpha = 1.0 / double(num_mag_sym());

    for (int i = 0; i < num_mag_sym(); i++) {
        /* full space-group symmetry operation is {R|t} */
        int pr = magnetic_group_symmetry(i).spg_op.proper;
        auto eang = magnetic_group_symmetry(i).spg_op.euler_angles;
        int isym = magnetic_group_symmetry(i).isym;
        auto S = magnetic_group_symmetry(i).spin_rotation;
        SHT::rotation_matrix(lmax, eang, pr, rotm);

        for (int ia = 0; ia < num_atoms_; ia++) {
            int ja = sym_table_(ia, isym);
            auto location = spl_atoms.location(ja);
            if (location.rank == comm__.rank()) {
                linalg<CPU>::gemm(0, 0, lmmax, nrmax, lmmax, alpha * S(2, 2), rotm.at<CPU>(), rotm.ld(),
                                  vz_rlm__.at<CPU>(0, 0, ia), vz_rlm__.ld(), 1.0,
                                  fsym.at<CPU>(0, 0, location.local_index), fsym.ld());
            }
        }
    }

    double* sbuf = spl_atoms.local_size() ? fsym.at<CPU>() : nullptr;
    comm__.allgather(sbuf, vz_rlm__.at<CPU>(),
                     lmmax * nrmax * spl_atoms.global_offset(),
                     lmmax * nrmax * spl_atoms.local_size());
}

inline void Unit_cell_symmetry::symmetrize_vector_function(mdarray<double, 3>& vx_rlm__,
                                                 mdarray<double, 3>& vy_rlm__,
                                                 mdarray<double, 3>& vz_rlm__,
                                                 Communicator const& comm__) const
{
    PROFILE("sirius::Unit_cell_symmetry::symmetrize_vector_function_mt");

    int lmmax = (int)vx_rlm__.size(0);
    int nrmax = (int)vx_rlm__.size(1);

    splindex<block> spl_atoms(num_atoms_, comm__.size(), comm__.rank());

    int lmax = Utils::lmax_by_lmmax(lmmax);

    mdarray<double, 2> rotm(lmmax, lmmax);

    mdarray<double, 4> v_sym(lmmax, nrmax, spl_atoms.local_size(), 3);
    v_sym.zero();

    mdarray<double, 3> vtmp(lmmax, nrmax, 3);

    double alpha = 1.0 / double(num_mag_sym());

    std::vector<mdarray<double, 3>*> vrlm({&vx_rlm__, &vy_rlm__, &vz_rlm__});

    for (int i = 0; i < num_mag_sym(); i++) {
        /* full space-group symmetry operation is {R|t} */
        int pr = magnetic_group_symmetry(i).spg_op.proper;
        auto eang = magnetic_group_symmetry(i).spg_op.euler_angles;
        int isym = magnetic_group_symmetry(i).isym;
        auto S = magnetic_group_symmetry(i).spin_rotation;
        SHT::rotation_matrix(lmax, eang, pr, rotm);

        for (int ia = 0; ia < num_atoms_; ia++) {
            int ja = sym_table_(ia, isym);
            auto location = spl_atoms.location(ja);
            if (location.rank == comm__.rank()) {
                for (int k: {0, 1, 2}) {
                    linalg<CPU>::gemm(0, 0, lmmax, nrmax, lmmax, alpha, rotm.at<CPU>(), rotm.ld(),
                                      vrlm[k]->at<CPU>(0, 0, ia), vrlm[k]->ld(), 0.0,
                                      vtmp.at<CPU>(0, 0, k), vtmp.ld());
                }
                #pragma omp parallel
                for (int k: {0, 1, 2}) {
                    for (int j: {0, 1, 2}) {
                        #pragma omp for
                        for (int ir = 0; ir < nrmax; ir++) {
                            for (int lm = 0; lm < lmmax; lm++) {
                                v_sym(lm, ir, location.local_index, k) += S(k, j) * vtmp(lm, ir, j);
                            }
                        }
                    }
                }
            }
        }
    }

    for (int k: {0, 1, 2}) {
        double* sbuf = spl_atoms.local_size() ? v_sym.at<CPU>(0, 0, 0, k) : nullptr;
        comm__.allgather(sbuf, vrlm[k]->at<CPU>(),
                         lmmax * nrmax * spl_atoms.global_offset(),
                         lmmax * nrmax * spl_atoms.local_size());
    }
}

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
