/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __XC_POINT_FIELDS_HPP__
#define __XC_POINT_FIELDS_HPP__

#include "core/memory.hpp"
#include "core/r3/r3.hpp"

namespace sirius {

/// Physical primitive fields, or their raw discrete energy covectors.
/** One spin component denotes the total field, two denote up/down fields.
 *  Model quadrature weights never multiply the physical input fields.
 */
struct xc_point_fields_t
{
    mdarray<double, 2> rho;
    mdarray<double, 3> gradient;
    mdarray<double, 2> tau;

    xc_point_fields_t(int nr__, int ns__)
        : rho({nr__, ns__})
        , gradient({nr__, 3, ns__})
        , tau({nr__, ns__})
    {
    }
};

/// Raw geometry covectors at fixed electronic coefficients and radial fields.
/** Points are local Cartesian derivatives. Atom derivatives (3,natom) and the
 *  lattice derivative are globally summed, with Cartesian atom positions held
 *  fixed for the latter. Lattice vectors are columns. Neither forces nor stress
 *  signs/normalization are included, nor derivatives of model quadrature weights.
 */
struct xc_geometry_adjoint_t
{
    std::vector<r3::vector<double>> points;
    mdarray<double, 2> atoms;
    r3::matrix<double> lattice;

    xc_geometry_adjoint_t(int npoint__, int natom__)
        : points(npoint__)
        , atoms({3, natom__})
    {
        atoms.zero();
    }
};

} // namespace sirius

#endif
