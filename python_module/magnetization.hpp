/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <list>
#include <vector>
#include <tuple>
#include <sstream>
#include "density/density.hpp"
#include "context/simulation_context.hpp"

namespace sirius {

std::vector<double>
magnetization(Density& density)
{
    std::vector<double> lm(3, 0.0);
    auto result = density.get_magnetisation();

    for (int i = 0; i < 3; ++i) {
        lm[i] = result[i].total;
    }

    return lm;
}

std::string
sprint_magnetization(K_point_set& kset, const Density& density)
{
    auto& ctx       = kset.ctx();
    auto& unit_cell = kset.unit_cell();

    auto result_mag = density.get_magnetisation();
    std::stringstream sstream;

    char buffer[20000];

    if (ctx.num_mag_dims()) {
        std::sprintf(buffer, "atom              moment                |moment|");
        sstream << buffer;
        std::sprintf(buffer, "\n");
        sstream << buffer;
        for (int i = 0; i < 80; i++) {
            std::sprintf(buffer, "-");
            sstream << buffer;
        }
        std::sprintf(buffer, "\n");
        sstream << buffer;

        for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
            r3::vector<double> v({result_mag[0].mt[ia], result_mag[1].mt[ia], result_mag[2].mt[ia]});
            std::sprintf(buffer, "%4i  [%8.4f, %8.4f, %8.4f]  %10.6f", ia, v[0], v[1], v[2], v.length());
            sstream << buffer;
            std::sprintf(buffer, "\n");
            sstream << buffer;
        }

        std::sprintf(buffer, "\n");
        sstream << buffer;
    }

    return sstream.str();
}

} // namespace sirius
