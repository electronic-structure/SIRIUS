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

#if defined(__APEX)
#include <apex_api.hpp>
#endif
#include <omp.h>
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
#if defined(__APEX)
    apex::profiler* apex_p_;
#endif
    /// List of child timers that we called inside another timer.
    static std::vector<std::string>& stack()
    {
        static std::vector<std::string> stack_;
        return stack_;
    }

    /// Mapping between timer label and timer counters.
    static std::map<std::string, timer_stats_t>& timer_values()
    {
        static std::map<std::string, timer_stats_t> timer_values_;
        return timer_values_;
    }

    /// Return a reference to values of a given timer.
    static timer_stats_t& timer_values(std::string label__)
    {
        return timer_values()[label__];
    }

    /// Mapping between parent timer and child timers.
    /** This map is needed to build a call tree of timers with the information about "self" time
        and time spent in calling other timers. */
    static std::map<std::string, std::map<std::string, double>>& timer_values_ex()
    {
        /* the following map is stored:

           parent_timer_label1  |--- child_timer_label1, time1a
                                |--- child timer_label2, time2
                                |--- child_timer_label3, time3

           parent_timer_label2  |--- child_timer_label1, time1b
                                |--- child_timer_label4, time4

           etc.
        */
        static std::map<std::string, std::map<std::string, double>> timer_values_ex_;
        return timer_values_ex_;
    }

    /// Keep track of the starting time.
    inline static time_point_t& global_starting_time()
    {
        static bool initialized{false};
        static time_point_t t_;
        /* record the starting time */
        if (!initialized) {
            t_ = std::chrono::high_resolution_clock::now();
            initialized = true;
        }
        return t_;
    }

    timer(timer const& src) = delete;
    timer& operator=(timer const& src) = delete;

  public:

    /// Constructor.
    timer(std::string label__)
        : label_(label__)
        , active_(true)
    {
        /* measure the starting time */
        starting_time_ = std::chrono::high_resolution_clock::now();
        /* add timer label to the list of called timers */
        stack().push_back(label_);
#if defined(__APEX)
        apex_p_ = apex::start(label_);
#endif
    }

    /// Destructor.
    ~timer()
    {
        stop();
    }

    /// Move asigment operator.
    timer(timer&& src__)
    {
        this->label_         = src__.label_;
        this->starting_time_ = src__.starting_time_;
        this->active_        = src__.active_;
        src__.active_        = false;
#if defined(__APEX)
        this->apex_p_        = src__.apex_p_;
#endif
    }

    /// Stop the timer and update the statistics.
    double stop()
    {
        if (!active_) {
            return 0;
        }

        /* remove this timer name from the list; now last element contains
           the name of the parent timer */
        stack().pop_back();

        /* measure the time difference */
        auto t2    = std::chrono::high_resolution_clock::now();
        auto tdiff = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - starting_time_);
        double val = tdiff.count();

        auto& ts = timer_values(label_);
#ifdef __TIMER_SEQUENCE
        ts.sequence.push_back(starting_time_);
        ts.sequence.push_back(t2);
#endif
        ts.min_val = std::min(ts.min_val, val);
        ts.max_val = std::max(ts.max_val, val);
        ts.tot_val += val;
        ts.count++;

        if (stack().size() != 0) {
            /* last element contains the name of the parent timer */
            auto parent_label = stack().back();
            /* add value to the parent timer */
            if (timer_values_ex().count(parent_label) == 0) {
                timer_values_ex()[parent_label] = std::map<std::string, double>();
            }
            if (timer_values_ex()[parent_label].count(label_) == 0) {
                timer_values_ex()[parent_label][label_] = 0;
            }
            timer_values_ex()[parent_label][label_] += val;
        }
#if defined(__APEX)
        apex::stop(apex_p_);
#endif
        active_ = false;
        return val;
    }

    /// Print the timer statistics.
    static void print()
    {
        for (int i = 0; i < 140; i++) {
            printf("-");
        }
        printf("\n");
        printf("name                                                                 count      total        min        max    average    self (%%)\n");
        for (int i = 0; i < 140; i++) {
            printf("-");
        }
        printf("\n");
        for (auto& it: timer_values()) {

            double te{0};
            if (timer_values_ex().count(it.first)) {
                for (auto& it2: timer_values_ex()[it.first]) {
                    te += it2.second;
                }
            }
            if (te > it.second.tot_val) {
                printf("wrong timer values: %f %f\n", te, it.second.tot_val);
                throw std::runtime_error("terminating...");

            }
            printf("%-65s : %6i %10.4f %10.4f %10.4f %10.4f     %6.2f\n", it.first.c_str(),
                                                                          it.second.count,
                                                                          it.second.tot_val,
                                                                          it.second.min_val,
                                                                          it.second.max_val,
                                                                          it.second.tot_val / it.second.count,
                                                                          (it.second.tot_val - te) / it.second.tot_val * 100);
        }
    }

    static void print_tree()
    {
        if (!timer_values().count(main_timer_label)) {
            return;
        }
        for (int i = 0; i < 140; i++) {
            printf("-");
        }
        printf("\n");

        double ttot = timer_values()[main_timer_label].tot_val;

        for (auto& it: timer_values()) {
            if (timer_values_ex().count(it.first)) {
                /* collect external times */
                double te{0};
                for (auto& it2: timer_values_ex()[it.first]) {
                    te += it2.second;
                }
                double f = it.second.tot_val / ttot;
                if (f > 0.01) {
                    printf("%s (%10.4fs, %.2f %% of self, %.2f %% of total)\n",
                           it.first.c_str(), it.second.tot_val, (it.second.tot_val - te) / it.second.tot_val * 100, f * 100);

                    std::vector<std::pair<double, std::string>> tmp;

                    for (auto& it2: timer_values_ex()[it.first]) {
                        tmp.push_back(std::pair<double, std::string>(it2.second / it.second.tot_val, it2.first));
                    }
                    std::sort(tmp.rbegin(), tmp.rend());
                    for (auto& e: tmp) {
                        printf("|--%s (%10.4fs, %.2f %%) \n", e.second.c_str(), timer_values_ex()[it.first][e.second], e.first * 100);
                    }
                }
            }
        }
    }

    static nlohmann::json serialize()
    {
        nlohmann::json dict;

        /* collect local timers */
        for (auto& it: timer::timer_values()) {
            timer_stats_t ts;
            nlohmann::json node;
            node["count"] = it.second.count;
            node["total"] = it.second.tot_val;
            node["min"]   = it.second.min_val;
            node["max"]   = it.second.max_val;
            node["avg"]   = it.second.tot_val / it.second.count;
#ifdef __TIMER_SEQUENCE
            std::vector<double> tseq;
            for (auto& s: it.second.sequence) {
                auto tdiff = std::chrono::duration_cast<std::chrono::duration<double>>(s - global_starting_time());
                tseq.push_back(tdiff.count());
            }
            node["sequence"] = tseq;
#endif
            dict[it.first] = node;
        }
        return std::move(dict);
    }

    static nlohmann::json serialize_tree()
    {
        nlohmann::json dict;

        if (!timer_values().count(main_timer_label)) {
            return {};
        }
        /* total execution time */
        double ttot = timer_values()[main_timer_label].tot_val;

        /* loop over the timer; iterator `it` is a <key, valu> pair */
        for (auto& it: timer_values()) {
            /* if this timer is a parent timer for somebody and timer has a non-negligible contribution */
            if (timer_values_ex().count(it.first) && (it.second.tot_val / ttot) > 0.01) {
                /* collect child (external) times */
                double te{0};
                for (auto& it2: timer_values_ex()[it.first]) {
                    te += it2.second;
                }
                nlohmann::json node;

                node["total_time"]              = it.second.tot_val;
                node["child_time"]              = te;
                node["self_time"]               = it.second.tot_val - te;
                node["fraction_of_global_time"] = it.second.tot_val / ttot;
                node["call"] = {};

                /* add all children values */
                for (auto& it2: timer_values_ex()[it.first]) {
                    nlohmann::json n;
                    n["time"]               = timer_values_ex()[it.first][it2.first];
                    n["fraction_of_parent"] = it2.second / it.second.tot_val;
                    node["call"][it2.first] = n;
                }
                dict[it.first] = node;
            }
        }

        return std::move(dict);
    }

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
    timer::global_timer();
}

/// Stops the global timer.
/** When global timer is stoped the timer tree can be build using timer::serialize_timers_tree() method. */
inline void stop_global_timer()
{
    timer::global_timer().stop();
}

}

#endif
