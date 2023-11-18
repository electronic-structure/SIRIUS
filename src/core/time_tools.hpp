// Copyright (c) 2013-2023 Anton Kozhevnikov, Thomas Schulthess
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

/** \file time_tools.hpp
 *
 *  \brief Timing functions.
 */

#ifndef __TIME_TOOLS_HPP__
#define __TIME_TOOLS_HPP__

#include <sys/time.h>

namespace sirius {

/// Return the timestamp string in a specified format.
/** Typical format strings: "%Y%m%d_%H%M%S", "%Y-%m-%d %H:%M:%S", "%H:%M:%S" */
inline auto
timestamp(std::string fmt)
{
    timeval t;
    gettimeofday(&t, NULL);

    char buf[128];

    tm* ptm = localtime(&t.tv_sec);
    strftime(buf, sizeof(buf), fmt.c_str(), ptm);
    return std::string(buf);
}

/// Wall-clock time in seconds.
inline double
wtime()
{
    timeval t;
    gettimeofday(&t, NULL);
    return double(t.tv_sec) + double(t.tv_usec) / 1e6;
}

using time_point_t = std::chrono::high_resolution_clock::time_point;

inline auto
time_now()
{
    return std::chrono::high_resolution_clock::now();
}

inline double
time_interval(std::chrono::high_resolution_clock::time_point t0)
{
    return std::chrono::duration_cast<std::chrono::duration<double>>(time_now() - t0).count();
}

} // namespace sirius

#endif
