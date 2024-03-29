/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file symmetrize_stress_tensor.hpp
 *
 *  \brief Symmetrize lattice stress tensor.
 */

#ifndef __SYMMETRIZE_STRESS_TENSOR_HPP__
#define __SYMMETRIZE_STRESS_TENSOR_HPP__

#include "crystal_symmetry.hpp"

namespace sirius {

inline void
symmetrize_stress_tensor(Crystal_symmetry const& sym__, r3::matrix<double>& s__)
{
    if (sym__.size() == 1) {
        return;
    }

    r3::matrix<double> result;

    for (int i = 0; i < sym__.size(); i++) {
        auto R = sym__[i].spg_op.Rcp;
        result = result + dot(dot(transpose(R), s__), R);
    }

    s__ = result * (1.0 / sym__.size());

    std::vector<std::array<int, 2>> idx = {{0, 1}, {0, 2}, {1, 2}};
    for (auto e : idx) {
        s__(e[0], e[1]) = s__(e[1], e[0]) = 0.5 * (s__(e[0], e[1]) + s__(e[1], e[0]));
    }
}

} // namespace sirius

#endif
