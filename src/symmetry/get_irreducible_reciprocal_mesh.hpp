/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file get_irreducible_reciprocal_mesh.hpp
 *
 *  \brief Find the irriducible k-points of the Brillouin zone.
 */

#ifndef __GET_IRREDUCIBLE_RECIPROCAL_MESH_HPP__
#define __GET_IRREDUCIBLE_RECIPROCAL_MESH_HPP__

#include "symmetry/crystal_symmetry.hpp"

namespace sirius {

inline std::tuple<int, std::vector<double>, std::vector<std::array<double, 3>>>
get_irreducible_reciprocal_mesh(Crystal_symmetry const& sym__, r3::vector<int> k_mesh__, r3::vector<int> is_shift__)
{
    using M = std::array<std::array<int, 3>, 3>;
    std::map<M, int> sym_map;

    for (int isym = 0; isym < sym__.size(); isym++) {
        M s;
        for (int x : {0, 1, 2}) {
            for (int y : {0, 1, 2}) {
                s[x][y] = sym__[isym].spg_op.R(x, y);
            }
        }
        sym_map[s] = 1;
    }
    std::vector<M> sym_list;
    for (auto it = sym_map.begin(); it != sym_map.end(); it++) {
        sym_list.push_back(it->first);
    }

    int nktot = k_mesh__[0] * k_mesh__[1] * k_mesh__[2];

    mdarray<int, 2> ikgrid({3, nktot});
    std::vector<int> ikmap(nktot, 0);

    double q[] = {0, 0, 0};
    int nk = spg_get_stabilized_reciprocal_mesh((int(*)[3]) & ikgrid(0, 0), &ikmap[0], &k_mesh__[0], &is_shift__[0], 1,
                                                static_cast<int>(sym_list.size()), (int(*)[3][3]) & sym_list[0], 1,
                                                (double(*)[3])(&q[0]));

    std::map<int, int> ikcount;
    for (int ik = 0; ik < nktot; ik++) {
        if (ikcount.count(ikmap[ik]) == 0) {
            ikcount[ikmap[ik]] = 0;
        }
        ikcount[ikmap[ik]]++;
    }

    std::vector<double> wk(nk);
    std::vector<std::array<double, 3>> kp(nk);

    int n{0};
    for (auto it = ikcount.begin(); it != ikcount.end(); it++) {
        wk[n] = static_cast<double>(it->second) / nktot;
        for (int x : {0, 1, 2}) {
            kp[n][x] = (ikgrid(x, it->first) + is_shift__[x] / 2.0) / k_mesh__[x];
        }
        n++;
    }
    if (n != nk) {
        RTE_THROW("wrong number of k-points");
    }

    return std::make_tuple(nk, wk, kp);
}

} // namespace sirius

#endif
