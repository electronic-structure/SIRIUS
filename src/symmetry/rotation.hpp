/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file rotation.hpp
 *
 *  \brief Generate rotation matrices and related entities.
 */

#ifndef __ROTATION_HPP__
#define __ROTATION_HPP__

#include "core/memory.hpp"
#include "core/r3/r3.hpp"
#include "core/rte/rte.hpp"
#include "core/constants.hpp"

namespace sirius {

/// Return angle phi in the range [0, 2Pi) by its values of sin(phi) and cos(phi).
inline double
phi_by_sin_cos(double sinp, double cosp)
{
    double phi = std::atan2(sinp, cosp);
    if (phi < 0) {
        phi += twopi;
    }
    return phi;
}

/// Generate SU(2) rotation matrix from the axes and angle.
inline auto
rotation_matrix_su2(std::array<double, 3> u__, double theta__)
{
    mdarray<std::complex<double>, 2> rotm({2, 2});

    auto cost = std::cos(theta__ / 2);
    auto sint = std::sin(theta__ / 2);

    rotm(0, 0) = std::complex<double>(cost, -u__[2] * sint);
    rotm(1, 1) = std::complex<double>(cost, u__[2] * sint);
    rotm(0, 1) = std::complex<double>(-u__[1] * sint, -u__[0] * sint);
    rotm(1, 0) = std::complex<double>(u__[1] * sint, -u__[0] * sint);

    return rotm;
}

/// Generate SU2(2) rotation matrix from a 3x3 rotation matrix in Cartesian coordinates.
/** Create quaternion components from the 3x3 matrix. The components are just a w = Cos(\Omega/2)
 *  and {x,y,z} = unit rotation vector multiplied by Sin(\Omega/2)
 *
 *  See https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
 *  and https://en.wikipedia.org/wiki/Rotation_group_SO(3)#Quaternions_of_unit_norm */
inline auto
rotation_matrix_su2(r3::matrix<double> R__)
{
    double det = R__.det() > 0 ? 1.0 : -1.0;

    r3::matrix<double> mat = R__ * det;

    mdarray<std::complex<double>, 2> su2mat({2, 2});

    su2mat.zero();

    /* make quaternion components*/
    double w = std::sqrt(std::max(0.0, 1.0 + mat(0, 0) + mat(1, 1) + mat(2, 2))) / 2.0;
    double x = std::sqrt(std::max(0.0, 1.0 + mat(0, 0) - mat(1, 1) - mat(2, 2))) / 2.0;
    double y = std::sqrt(std::max(0.0, 1.0 - mat(0, 0) + mat(1, 1) - mat(2, 2))) / 2.0;
    double z = std::sqrt(std::max(0.0, 1.0 - mat(0, 0) - mat(1, 1) + mat(2, 2))) / 2.0;

    x = std::copysign(x, mat(2, 1) - mat(1, 2));
    y = std::copysign(y, mat(0, 2) - mat(2, 0));
    z = std::copysign(z, mat(1, 0) - mat(0, 1));

    su2mat(0, 0) = std::complex<double>(w, -z);
    su2mat(1, 1) = std::complex<double>(w, z);
    su2mat(0, 1) = std::complex<double>(-y, -x);
    su2mat(1, 0) = std::complex<double>(y, -x);

    return su2mat;
}

/// Get axis and angle from rotation matrix.
inline auto
axis_angle(r3::matrix<double> R__)
{
    r3::vector<double> u;
    /* make proper rotation */
    R__  = R__ * R__.det();
    u[0] = R__(2, 1) - R__(1, 2);
    u[1] = R__(0, 2) - R__(2, 0);
    u[2] = R__(1, 0) - R__(0, 1);

    double sint = u.length() / 2.0;
    double cost = (R__(0, 0) + R__(1, 1) + R__(2, 2) - 1) / 2.0;

    double theta = phi_by_sin_cos(sint, cost);

    /* rotation angle is zero */
    if (std::abs(theta) < 1e-12) {
        u = {0, 0, 1};
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

    return std::pair<r3::vector<double>, double>(u, theta);
}

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
inline auto
rot_mtrx_cart(r3::vector<double> euler_angles__)
{
    double alpha = euler_angles__[0];
    double beta  = euler_angles__[1];
    double gamma = euler_angles__[2];

    r3::matrix<double> rm;
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

/// Compute Euler angles corresponding to the proper rotation matrix.
inline auto
euler_angles(r3::matrix<double> const& rot__, double tolerance__)
{
    r3::vector<double> angles(0, 0, 0);

    if (std::abs(rot__.det() - 1) > 1e-10) {
        std::stringstream s;
        s << "determinant of rotation matrix is " << rot__.det();
        RTE_THROW(s);
    }

    auto rit = inverse(transpose(rot__));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (std::abs(rot__(i, j) - rit(i, j)) > tolerance__) {
                std::stringstream s;
                s << "rotation matrix is not unitary" << std::endl
                  << "initial symmetry matrix:" << std::endl
                  << rot__ << std::endl
                  << "inverse transpose matrix:" << std::endl
                  << rit;
                RTE_THROW(s);
            }
        }
    }

    if (std::abs(rot__(2, 2) - 1.0) < 1e-10) { // cos(beta) == 1, beta = 0
        angles[0] = phi_by_sin_cos(rot__(1, 0), rot__(0, 0));
    } else if (std::abs(rot__(2, 2) + 1.0) < 1e-10) { // cos(beta) == -1, beta = Pi
        angles[0] = phi_by_sin_cos(-rot__(0, 1), rot__(1, 1));
        angles[1] = pi;
    } else {
        double beta = std::acos(rot__(2, 2));
        angles[0]   = phi_by_sin_cos(rot__(1, 2) / std::sin(beta), rot__(0, 2) / std::sin(beta));
        angles[1]   = beta;
        angles[2]   = phi_by_sin_cos(rot__(2, 1) / std::sin(beta), -rot__(2, 0) / std::sin(beta));
    }

    auto rm1 = rot_mtrx_cart(angles);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (std::abs(rot__(i, j) - rm1(i, j)) > tolerance__) {
                std::stringstream s;
                s << "matrices don't match" << std::endl
                  << "initial symmetry matrix: " << std::endl
                  << rot__ << std::endl
                  << "euler angles : " << angles[0] / pi << " " << angles[1] / pi << " " << angles[2] / pi << std::endl
                  << "computed symmetry matrix : " << std::endl
                  << rm1;
                RTE_THROW(s);
            }
        }
    }

    return angles;
}

} // namespace sirius

#endif // __ROTATION_HPP__
