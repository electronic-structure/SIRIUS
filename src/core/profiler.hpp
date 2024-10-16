/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

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
#if defined(SIRIUS_GPU) && defined(SIRIUS_TX)
#include "core/acc/tx_profiler.hpp"
#endif

namespace sirius {

extern ::rt_graph::Timer global_rtgraph_timer;

#if defined(SIRIUS_TX)
extern acc::txprofiler::Timer global_tx_timer;
#endif

// TODO: add calls to apex

#if defined(SIRIUS_PROFILE)
#define PROFILER_CONCAT_IMPL(x, y) x##y
#define PROFILER_CONCAT(x, y) PROFILER_CONCAT_IMPL(x, y)

#if defined(SIRIUS_TX)
#define PROFILE(identifier)                                                                                            \
    acc::txprofiler::ScopedTiming PROFILER_CONCAT(GeneratedScopedTimer, __COUNTER__)(identifier, global_tx_timer);     \
    ::rt_graph::ScopedTiming PROFILER_CONCAT(GeneratedScopedTimer, __COUNTER__)(identifier, global_rtgraph_timer);
#define PROFILE_START(identifier)                                                                                      \
    global_tx_timer.start(identifier);                                                                                 \
    global_rtgraph_timer.start(identifier);
#define PROFILE_STOP(identifier)                                                                                       \
    global_rtgraph_timer.stop(identifier);                                                                             \
    global_tx_timer.stop(identifier);
#else /* NVTX and ROCTX are not defined -> just use rt_graph */
#define PROFILE(identifier)                                                                                            \
    ::rt_graph::ScopedTiming PROFILER_CONCAT(GeneratedScopedTimer, __COUNTER__)(identifier, global_rtgraph_timer);
#define PROFILE_START(identifier) global_rtgraph_timer.start(identifier);
#define PROFILE_STOP(identifier) global_rtgraph_timer.stop(identifier);
#endif // SIRIUS_CUDA_NVTX

#else
#define PROFILE(...)
#define PROFILE_START(...)
#define PROFILE_STOP(...)
#endif

} // namespace sirius

#endif
