// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file profiler.hpp
 *
 *  \brief A time-based profiler.
 */

#ifndef __PROFILER_HPP__
#define __PROFILER_HPP__

#include <mpi.h>
#include <string>
#if defined(__APEX)
#include <apex_api.hpp>
#endif
#include "core/rt_graph.hpp"
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA_NVTX)
#include "core/acc/nvtx_profiler.hpp"
#endif

namespace sirius {

extern ::rt_graph::Timer global_rtgraph_timer;

#if defined(SIRIUS_CUDA_NVTX)
extern acc::nvtxprofiler::Timer global_nvtx_timer;
#endif

// TODO: add calls to apex and cudaNvtx

#if defined(SIRIUS_PROFILE)
#define PROFILER_CONCAT_IMPL(x, y) x##y
#define PROFILER_CONCAT(x, y) PROFILER_CONCAT_IMPL(x, y)

#if defined(SIRIUS_CUDA_NVTX)
#define PROFILE(identifier)                                                                                            \
    acc::nvtxprofiler::ScopedTiming PROFILER_CONCAT(GeneratedScopedTimer, __COUNTER__)(identifier, global_nvtx_timer); \
    ::rt_graph::ScopedTiming PROFILER_CONCAT(GeneratedScopedTimer, __COUNTER__)(identifier, global_rtgraph_timer);
#define PROFILE_START(identifier)                                                                                      \
    global_nvtx_timer.start(identifier);                                                                               \
    global_rtgraph_timer.start(identifier);
#define PROFILE_STOP(identifier)                                                                                       \
    global_rtgraph_timer.stop(identifier);                                                                             \
    global_nvtx_timer.stop(identifier);
#else
#define PROFILE(identifier)                                                                                            \
    ::rt_graph::ScopedTiming PROFILER_CONCAT(GeneratedScopedTimer, __COUNTER__)(identifier, global_rtgraph_timer);
#define PROFILE_START(identifier) global_rtgraph_timer.start(identifier);
#define PROFILE_STOP(identifier) global_rtgraph_timer.stop(identifier);
#endif

#else
#define PROFILE(...)
#define PROFILE_START(...)
#define PROFILE_STOP(...)
#endif

} // namespace sirius

#endif
