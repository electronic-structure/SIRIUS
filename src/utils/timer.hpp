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

/** \file timer.hpp
 *
 *  \brief Implementation of utils::timer class.
 */

#ifndef __TIMER_HPP__
#define __TIMER_HPP__

//#define __TIMER_SEQUENCE

#include <string>
#include <sstream>
#include <chrono>
#include <map>
#include <vector>
#include <memory>
#include <complex>
#include <algorithm>
#include "json.hpp"

namespace utils {

/// Alias for the time point type.
using time_point_t = std::chrono::high_resolution_clock::time_point;

/// Timer statistics.
struct timer_stats_t
{
    /// Minimum value of the timer.
    double min_val{1e10};
    /// Maximum value of the timer.
    double max_val{0};
    /// Cumulitive time.
    double tot_val{0};
    /// Average time.
    double avg_val{0};
    /// Number of measurments.
    int count{0};
#ifdef __TIMER_SEQUENCE
    /// Full sequence of start and stop times.
    std::vector<time_point_t> sequence;
#endif
};

/// Name of the global timer.
const std::string main_timer_label = "+global_timer";

/// A simple timer implementation.
class timer
{
  private:
    /// Unique label of the timer.
    std::string label_;

    /// Starting time.
    time_point_t starting_time_;

    /// True if timer is active.
    bool active_{false};

    /// List of child timers that we called inside another timer.
    static std::vector<std::string>& stack();

    /// Mapping between timer label and timer counters.
    static std::map<std::string, timer_stats_t>& timer_values();

    /// Return a reference to values of a given timer.
    static timer_stats_t& timer_values(std::string label__)
    {
        return timer_values()[label__];
    }

    /// Mapping between parent timer and child timers.
    /** This map is needed to build a call tree of timers with the information about "self" time
        and time spent in calling other timers. */
    static std::map<std::string, std::map<std::string, double>>& timer_values_ex();

    /// Keep track of the starting time.
    static time_point_t& global_starting_time();

    timer(timer const& src) = delete;
    timer& operator=(timer const& src) = delete;

  public:

    /// Constructor.
    timer(std::string label__)
        : label_(label__)
        , active_(true)
    {
#if defined(__USE_TIMER)
        /* measure the starting time */
        starting_time_ = std::chrono::high_resolution_clock::now();
        /* add timer label to the list of called timers */
        stack().push_back(label_);
#endif
    }

    /// Destructor.
    ~timer()
    {
#if defined(__USE_TIMER)
        stop();
#endif
    }

    /// Move asigment operator.
    timer(timer&& src__);

    /// Stop the timer and update the statistics.
    double stop();

    /// Print the timer statistics.
    static void print();

    static void print_tree();

    static nlohmann::json serialize();

    static nlohmann::json serialize_tree();

    inline static timer& global_timer()
    {
        global_starting_time();
        static timer global_timer__(main_timer_label);
        return global_timer__;
    }

    /// List of timers created on the Fortran side.
    inline static std::map<std::string, timer>& ftimers()
    {
        static std::map<std::string, timer> ftimers__;
        return ftimers__;
    }
};

/// Triggers the creation of the global timer.
/** The global timer is the parent for all other timers. */
inline void start_global_timer()
{
#if defined(__USE_TIMER)
    timer::global_timer();
#endif
}

/// Stops the global timer.
/** When global timer is stoped the timer tree can be build using timer::serialize_timers_tree() method. */
inline void stop_global_timer()
{
#if defined(__USE_TIMER)
    timer::global_timer().stop();
#endif
}

}

#endif
