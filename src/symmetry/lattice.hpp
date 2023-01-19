// Copyright (c) 2013-2021 Anton Kozhevnikov, Thomas Schulthess
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

/** \file lattice.hpp
 *
 *  \brief Crystal lattice functions.
 */

#include <algorithm>
#include "linalg/r3.hpp"
#include "utils/rte.hpp"

#ifndef __LATTICE_HPP__
#define __LATTICE_HPP__

namespace sirius {

/// Compute a metric tensor.
inline auto
metric_tensor(r3::matrix<double> const& lat_vec__)
{
    return dot(transpose(lat_vec__), lat_vec__);
}

/// Compute error of the symmetry-transformed metric tensor.
inline double
metric_tensor_error(r3::matrix<double> const& lat_vec__, r3::matrix<int> const& R__)
{
    auto mt = metric_tensor(lat_vec__);

    double diff{0};
    auto mt1 = dot(dot(transpose(R__), mt), R__);
    for (int i: {0, 1, 2}) {
        for (int j: {0, 1, 2}) {
            diff = std::max(diff, std::abs(mt1(i, j) - mt(i, j)));
        }
    }
    return diff;
}

inline auto
find_lat_sym(r3::matrix<double> const& lat_vec__, double tol__, double* mt_error__ = nullptr)
{
    std::vector<r3::matrix<int>> lat_sym;

    auto r = {-1, 0, 1};

    if (mt_error__) {
        *mt_error__ = 0;
    }

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
                r3::matrix<int> R({{i00, i01, i02}, {i10, i11, i12}, {i20, i21, i22}});
                /* valid symmetry operation has a determinant of +/- 1 */
                if (std::abs(R.det()) == 1) {
                    auto mt_error = metric_tensor_error(lat_vec__, R);
                    /* metric tensor should be invariant under symmetry operation */
                    if (mt_error < tol__) {
                        lat_sym.push_back(R);
                        if (mt_error__) {
                            *mt_error__ = std::max(*mt_error__, mt_error);
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
    }

    if (lat_sym.size() == 0 || lat_sym.size() > 48) {
        std::stringstream s;
        s << "wrong number of lattice symmetries: " << lat_sym.size();
        RTE_THROW(s);
    }

    /* check if the set of symmetry operations is a group */
    for (auto& R1: lat_sym) {
        for (auto& R2: lat_sym) {
            auto R3 = r3::dot(R1, R2);
            if (std::find(lat_sym.begin(), lat_sym.end(), R3) == lat_sym.end()) {
                std::stringstream s;
                s << "lattice symmetries do not form a group" << std::endl;
                for (auto& R: lat_sym) {
                    s << " sym.op : " << R << ", metric tensor error : " << metric_tensor_error(lat_vec__, R) << std::endl;
                }
                s << "R1 : " << R1 << std::endl;
                s << "R2 : " << R2 << std::endl;
                s << "R1 * R2 : " << R3 << " is not in group" << std::endl;
                s << "metric tensor tolerance : " << tol__;
                RTE_THROW(s);
            }
        }
    }

    return lat_sym;
}

}

#endif
