/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __NVTX_PROFILER_HPP__
#define __NVTX_PROFILER_HPP__

#if defined(SIRIUS_CUDA_NVTX)

#include <unordered_map>
#include "nvToolsExt.h"

namespace sirius {

namespace acc {

namespace nvtxprofiler {

class Timer
{
  public:
    void
    start(std::string const& str)
    {
        timers_[str] = nvtxRangeStartA(str.c_str());
    }

    void
    stop(std::string const& str)
    {
        auto result = timers_.find(str);
        if (result == timers_.end())
            return;
        nvtxRangeEnd(result->second);
        timers_.erase(result);
    }

  private:
    std::unordered_map<std::string, nvtxRangeId_t> timers_;
};

class ScopedTiming
{
  public:
    ScopedTiming(std::string identifier, Timer& timer)
        : identifier_(identifier)
        , timer_(timer)
    {
        timer.start(identifier_);
    }

    ScopedTiming(const ScopedTiming&) = delete;
    ScopedTiming(ScopedTiming&&)      = delete;
    auto
    operator=(const ScopedTiming&) -> ScopedTiming& = delete;
    auto
    operator=(ScopedTiming&&) -> ScopedTiming& = delete;

    ~ScopedTiming()
    {
        timer_.stop(identifier_);
    }

  private:
    std::string identifier_;
    Timer& timer_;
};

} // namespace nvtxprofiler
} // namespace acc
} // namespace sirius
#endif
#endif
