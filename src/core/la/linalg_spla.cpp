/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <memory>
#include <spla/spla.hpp>
#include "linalg_spla.hpp"

namespace sirius {
namespace splablas {

std::shared_ptr<::spla::Context>&
get_handle_ptr()
{
    static std::shared_ptr<::spla::Context> handle{new ::spla::Context{SPLA_PU_HOST}};
    return handle;
}

} // namespace splablas
} // namespace sirius
