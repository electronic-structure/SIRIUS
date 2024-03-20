/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "memory.hpp"

namespace sirius {

/// Return a memory pool.
/** A memory pool is created when this function called for the first time. */
memory_pool&
get_memory_pool(memory_t M__)
{
    static std::map<memory_t, memory_pool> memory_pool_;
    if (memory_pool_.count(M__) == 0) {
        memory_pool_.emplace(M__, memory_pool(M__));
    }
    return memory_pool_.at(M__);
}

} // namespace sirius
