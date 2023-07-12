// Copyright (c) 2013-2023 Anton Kozhevnikov, Thomas Schulthess
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

}

#endif

