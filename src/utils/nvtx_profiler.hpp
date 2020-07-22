#ifndef NVTX_PROFILER_H
#define NVTX_PROFILER_H

#if defined(__CUDA_NVTX)

#include "nvToolsExt.h"
#include <unordered_map>

namespace nvtxprofiler {

class Timer {
public:
  void start(std::string const &str) {
    timers_[str] = nvtxRangeStartA(str.c_str());
  }

  void stop(std::string const &str) {
    auto result = timers_.find(str);
    if (result == timers_.end()) return;
    nvtxRangeEnd(result->second);
    timers_.erase(result);
  }

private:
  std::unordered_map<std::string, nvtxRangeId_t> timers_;
};

class ScopedTiming {
public:
  ScopedTiming(std::string identifier, Timer &timer) :
    identifier_(identifier), timer_(timer) {
    timer.start(identifier_);
  }

  ScopedTiming(const ScopedTiming&) = delete;
  ScopedTiming(ScopedTiming&&) = delete;
  auto operator=(const ScopedTiming&) -> ScopedTiming& = delete;
  auto operator=(ScopedTiming &&) -> ScopedTiming& = delete;

  ~ScopedTiming() {
    timer_.stop(identifier_);
  }

private:
  std::string identifier_;
  Timer& timer_;
};

}

#endif
#endif