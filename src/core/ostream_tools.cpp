/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file ostream_tools.cpp
 *
 *  \brief Output stream tools.
 */

#include "ostream_tools.hpp"

namespace sirius {

null_stream_t&
null_stream()
{
    static null_stream_t null_stream__;
    return null_stream__;
}

} // namespace sirius
