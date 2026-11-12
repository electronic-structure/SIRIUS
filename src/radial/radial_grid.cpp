/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file radial_grid.cpp
 *
 *  \brief Radial grid type from a string.
 */

#include <algorithm>
#include <map>
#include <sstream>
#include <stdexcept>
#include "radial/radial_grid.hpp"

namespace sirius {

std::pair<radial_grid_t, double>
get_radial_grid_t(std::string str__)
{
    auto pos = str__.find(",");
    if (pos == std::string::npos) {
        std::stringstream s;
        s << "wrong string for the radial grid type: " << str__;
        RTE_THROW(s);
    }

    std::string name = str__.substr(0, pos);
    double p         = std::stod(str__.substr(pos + 1));

    const std::map<std::string, radial_grid_t> map_to_type = {{"linear", radial_grid_t::linear},
                                                              {"exponential", radial_grid_t::exponential},
                                                              {"power", radial_grid_t::power},
                                                              {"lin_exp", radial_grid_t::lin_exp}};

    return std::make_pair(map_to_type.at(name), p);
}

} // namespace sirius
