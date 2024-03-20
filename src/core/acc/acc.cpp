/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file acc.cpp
 *
 *  \brief Definition of the functions for the acc:: namespace.
 *
 */
#include <atomic>
#include "acc.hpp"

namespace sirius {

namespace acc {

int
num_devices()
{
#if defined(SIRIUS_CUDA) || defined(SIRIUS_ROCM)
    static std::atomic<int> count(-1);
    if (count.load(std::memory_order_relaxed) == -1) {
        int c;
        if (GPU_PREFIX(GetDeviceCount)(&c) != GPU_PREFIX(Success)) {
            count.store(0, std::memory_order_relaxed);
        } else {
            count.store(c, std::memory_order_relaxed);
        }
    }
    return count.load(std::memory_order_relaxed);
#else
    return 0;
#endif
}

std::vector<acc_stream_t>&
streams()
{
    static std::vector<acc_stream_t> streams_;
    return streams_;
}

} // namespace acc

} // namespace sirius
