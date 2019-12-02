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
#include "timer.hpp"
#include "rt_graph.hpp"
#if defined(__GPU) && defined(__CUDA_NVTX)
#include "GPU/acc.hpp"
#endif

/* this are set by CMake */
//#define __PROFILE
//#define __PROFILE_TIME
//#define __PROFILE_STACK
//#define __PROFILE_FUNC

namespace utils {

/// Simple profiler and function call tracker.
class profiler
{
  private:
    /// Label of the profiler.
    std::string label_;

    /// Name of the function in which the profiler is created.
    std::string function_name_;

    /// Name of the file.
    std::string file_;

    /// Line number.
    int line_;

    /// Profiler's timer.
    std::unique_ptr<utils::timer> timer_;

#if defined(__APEX)
    apex::profiler* apex_p_;
#endif
#if defined(__PROFILE_STACK)
    static std::vector<std::string>& call_stack()
    {
        static std::vector<std::string> call_stack_;
        return call_stack_;
    }
#endif

  public:
    /// Constructor.
    profiler(char const* function_name__, char const* file__, int line__, char const* label__)
        : label_(std::string(label__))
        , function_name_(std::string(function_name__))
        , file_(std::string(file__))
        , line_(line__)
    {
#if defined(__PROFILE_STACK) || defined(__PROFILE_FUNC)
        char str[2048];
        snprintf(str, 2048, "%s at %s:%i", function_name__, file__, line__);
#endif

#if defined(__PROFILE_STACK)
        call_stack().push_back(std::string(str));
#endif

#if defined(__PROFILE_FUNC)
        int tab{0};
#if defined(__PROFILE_STACK)
        tab = static_cast<int>(call_stack().size()) - 1;
#endif
        for (int i = 0; i < tab; i++) {
            std::printf(" ");
        }
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::printf("[rank%04i] + %s\n", rank, label_.c_str());
#endif

#if defined(__PROFILE_TIME)
        timer_ = std::unique_ptr<utils::timer>(new utils::timer(label_));
#endif

#if defined(__GPU) && defined(__CUDA_NVTX)
        acc::begin_range_marker(label_.c_str());
#endif
#if defined(__APEX)
        apex_p_ = apex::start(label_);
#endif
    }

    ~profiler()
    {
#ifdef __PROFILE_FUNC
        int tab{0};
#ifdef __PROFILE_STACK
        tab = static_cast<int>(call_stack().size()) - 1;
#endif
        for (int i = 0; i < tab; i++) {
            std::printf(" ");
        }
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::printf("[rank%04i] - %s\n", rank, label_.c_str());
#endif

#ifdef __PROFILE_STACK
        call_stack().pop_back();
#endif

#if defined(__GPU) && defined(__CUDA_NVTX)
        acc::end_range_marker();
#endif
#if defined(__APEX)
        apex::stop(apex_p_);
#endif
    }

    static void stack_trace()
    {
#ifdef __PROFILE_STACK
        int t{0};
        for (auto it = call_stack().rbegin(); it != call_stack().rend(); it++) {
            for (int i = 0; i < t; i++) {
                std::printf(" ");
            }
            std::printf("[%s]\n", it->c_str());
            t++;
        }
#endif
    }
};

#ifdef __GNUC__
    #define __function_name__ __PRETTY_FUNCTION__
#else
    #define __function_name__ __func__
#endif

// --------------------
// Profile with RTGraph
// --------------------

extern ::rt_graph::Timer global_rtgraph_timer;

#ifdef __PROFILE
#define PROFILER_CONCAT_IMPL(x, y) x##y
#define PROFILER_CONCAT(x, y) PROFILER_CONCAT_IMPL(x, y)

#define PROFILE(identifier)                                                                                            \
    ::rt_graph::ScopedTiming PROFILER_CONCAT(GeneratedScopedTimer, __COUNTER__)(identifier,                            \
                                                                                ::utils::global_rtgraph_timer);

#define PROFILE_START(identifier) ::utils::global_rtgraph_timer.start(identifier);
#define PROFILE_STOP(identifier) ::utils::global_rtgraph_timer.stop(identifier);

#else
#define PROFILE(...)
#define PROFILE_START(...)
#define PROFILE_STOP(...)
#endif

} // namespace utils

#endif
