/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __XC_POINT_FIELDS_HPP__
#define __XC_POINT_FIELDS_HPP__

#include "core/memory.hpp"

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

} // namespace sirius

#endif
