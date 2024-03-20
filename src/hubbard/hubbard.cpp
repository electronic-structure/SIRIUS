/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "hubbard.hpp"

namespace sirius {

Hubbard::Hubbard(Simulation_context& ctx__)
    : ctx_(ctx__)
    , unit_cell_(ctx__.unit_cell())
{
    if (!ctx_.hubbard_correction()) {
        return;
    }
}

} // namespace sirius
