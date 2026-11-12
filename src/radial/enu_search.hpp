/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file enu_search.hpp
 *
 *  \brief Parameters of the search for the linearization energy of a radial function.
 */

#ifndef __ENU_SEARCH_HPP__
#define __ENU_SEARCH_HPP__

namespace sirius {

struct enu_search_t
{
    double enu{0.0};
    double ebot{0.0};
    double etop{0.0};
    int auto_enu{1};
    double etop_first_pass{0};
};

} // namespace sirius

#endif
