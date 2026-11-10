/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file typedefs.cpp
 *
 *  \brief Conversion of enumerators to / from strings.
 */

#include <algorithm>
#include <map>
#include <sstream>
#include <stdexcept>
#include "core/typedefs.hpp"

namespace sirius {

relativity_t
get_relativity_t(std::string name__)
{
    std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);
    std::map<std::string, relativity_t> const m = {{"none", relativity_t::none},
                                                   {"koelling_harmon", relativity_t::koelling_harmon},
                                                   {"zora", relativity_t::zora},
                                                   {"iora", relativity_t::iora},
                                                   {"dirac", relativity_t::dirac}};

    if (m.count(name__) == 0) {
        std::stringstream s;
        s << "get_relativity_t(): wrong label of the relativity_t enumerator: " << name__;
        throw std::runtime_error(s.str());
    }
    return m.at(name__);
}

std::string
get_relativity(relativity_t rel__)
{
    std::map<relativity_t, std::string> const m = {{relativity_t::none, "none"},
                                                   {relativity_t::koelling_harmon, "Koelling-Harmon"},
                                                   {relativity_t::zora, "zora"},
                                                   {relativity_t::iora, "iora"},
                                                   {relativity_t::dirac, "Dirac"}};

    if (m.count(rel__) == 0) {
        std::stringstream s;
        s << "get_relativity(): wrong type of the relativity_t enumerator";
        throw std::runtime_error(s.str());
    }
    return m.at(rel__);
}

} // namespace sirius
