/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

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

inline double
time_interval(std::chrono::high_resolution_clock::time_point t0, std::chrono::high_resolution_clock::time_point t1)
{
    return std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
}

} // namespace sirius

#endif
