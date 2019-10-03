// Copyright (c) 2013-2019 Anton Kozhevnikov, Thomas Schulthess
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

/** \file find_lat_sym.hpp
 *
 *  \brief Find the lattice symmetries
 */

#include "geometry3d.hpp"

#ifndef __FIND_LAT_SYM_HPP__
#define __FIND_LAT_SYM_HPP__

namespace sirius {

using namespace geometry3d;

inline std::vector<matrix3d<int>> find_lat_sym(matrix3d<double> lat_vec__, double tol__)
{
    std::vector<matrix3d<int>> lat_sym;

    /* metric tensor */
    auto mt = transpose(lat_vec__) * lat_vec__;
    auto r = {-1, 0, 1};

    for (int i00: r) {
    for (int i01: r) {
    for (int i02: r) {
        for (int i10: r) {
        for (int i11: r) {
        for (int i12: r) {
            for (int i20: r) {
            for (int i21: r) {
            for (int i22: r) {
                /* build a trial symmetry operation */
                matrix3d<int> S({{i00, i01, i02}, {i10, i11, i12}, {i20, i21, i22}});
                /* valid symmetry operation has a determinant of +/- 1 */
                if (std::abs(S.det()) == 1) {
                    /* metric tensor should be invariant under symmetry operation */
                    auto mt1 = transpose(S) * mt * S;
                    double diff{0};
                    for (int i: {0, 1, 2}) {
                        for (int j: {0, 1, 2}) {
                            diff = std::max(diff, std::abs(mt1(i, j) - mt(i, j)));
                        }
                    }
                    if (diff < tol__) {
                        lat_sym.push_back(S);
                    }
                }
            }
            }
            }
        }
        }
        }
    }
    }
    }

    if (lat_sym.size() == 0 || lat_sym.size() > 48) {
        std::stringstream s;
        s << "find_lat_sym(): wrong number of lattice symmetries: " << lat_sym.size() << "\n";
        throw std::runtime_error(s.str());
    }

    return lat_sym;
}

}

#endif
