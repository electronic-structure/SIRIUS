#ifndef __TX_PROFILER_HPP__
#define __TX_PROFILER_HPP__

#if defined(SIRIUS_TX)
#include <unordered_map>
#include <string>

#if defined(SIRIUS_CUDA)
#include "nvToolsExt.h"
#endif

#if defined(SIRIUS_ROCM)
#include "roctx.h"
#endif

namespace sirius {

namespace acc {

namespace txprofiler {
/* roctx and nvtx ns */

enum class _vendor
{
    cuda,
    rocm
};

template <enum _vendor>
class TimerVendor
{
};

#if defined(SIRIUS_CUDA)
template <>
class TimerVendor<_vendor::cuda>
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
using timer_vendor_t = TimerVendor<_vendor::cuda>;
#endif /* SIRIUS_CUDA_NVTX */

#if defined(SIRIUS_ROCM)
template <>
class TimerVendor<_vendor::rocm>
{
  public:
    void
    start(std::string const& str)
    {
        timers_[str] = roctxRangeStartA(str.c_str());
    }

    void
    stop(std::string const& str)
    {
        auto result = timers_.find(str);
        if (result == timers_.end())
            return;
        roctxRangeStop(result->second);
        timers_.erase(result);
    }

  private:
    std::unordered_map<std::string, roctx_range_id_t> timers_;
};
using timer_vendor_t = TimerVendor<_vendor::rocm>;
#endif /* SIRIUS_ROCTX */

class Timer : timer_vendor_t
{
  public:
    void
    start(std::string const& str)
    {
        timer_vendor_t::start(str);
    }

    void
    stop(std::string const& str)
    {
        timer_vendor_t::stop(str);
    }
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

} // namespace txprofiler
} // namespace acc
} // namespace sirius
#endif /* defined(SIRIUS_CUDA_NVTX) || defined(SIRIUS_ROCTX) */
#endif /* __TX_PROFILER_HPP__ */
