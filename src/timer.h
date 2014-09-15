// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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

/** \file timer.h
 *   
 *  \brief Contains definition and partial implementation of sirius::Timer class.
 */

#ifndef __TIMER_H__
#define __TIMER_H__

#include <sys/time.h>
#include <map>
#include <string>
#include <iostream>
#include <vector>
#include "config.h"
#include "communicator.h"
#ifdef _TIMER_CHRONO_
#include <chrono>
#endif

namespace sirius 
{

const int _local_timer_ = 0;
const int _global_timer_ = 1;

struct timer_stats
{
    int count;
    double min_value;
    double max_value;
    double total_value;
    double average_value;
    int timer_type;
};

/// Simple timer interface.
class Timer
{
    private:
        
        /// string label of the timer
        std::string label_;
        
        /// starting time
        #if defined(_TIMER_TIMEOFDAY_)
        timeval starting_time_;
        #elif defined(_TIMER_MPI_WTIME_)
        double starting_time_;
        #elif defined(_TIMER_CHRONO_)
        std::chrono::high_resolution_clock::time_point starting_time_;
        #endif

        Communicator const* comm_;

        /// true if timer is running
        bool active_;

        int timer_type_;

        /// mapping between timer name and timer values
        static std::map< std::string, std::vector<double> > timers_;

        static std::map< std::string, std::vector<double> > global_timers_;
    
    public:
        
        Timer(std::string const& label__) 
            : label_(label__),
              comm_(nullptr),
              active_(false),
              timer_type_(_local_timer_)
        {
            if (timers_.count(label_) == 0) timers_[label_] = std::vector<double>();

            start();
        }

        Timer(std::string const& label__, int timer_type__) 
            : label_(label__),
              comm_(nullptr),
              active_(false),
              timer_type_(timer_type__)
        {
            switch (timer_type_)
            {
                case _local_timer_:
                {
                    if (timers_.count(label_) == 0) timers_[label_] = std::vector<double>();
                    break;
                }
                case _global_timer_:
                {
                    if (global_timers_.count(label_) == 0) global_timers_[label_] = std::vector<double>();
                    break;
                }
            }

            start();
        }

        Timer(std::string const& label__, Communicator const& comm__) 
            : label_(label__),
              comm_(&comm__),
              active_(false),
              timer_type_(_local_timer_)
        {
            if (timers_.count(label_) == 0) timers_[label_] = std::vector<double>();

            start();
        }

        ~Timer()
        {
            if (active_) stop();
        }

        void start();

        double stop();

        double value();

        static void clear()
        {
            timers_.clear();
        }

        static std::map< std::string, std::vector<double> >& timers()
        {
            return timers_;
        }

        static std::map< std::string, timer_stats> collect_timer_stats();

        static double value(std::string const& label__)
        {
            auto values = timers_[label__];

            double d = 0;
            for (int i = 0; i < (int)values.size(); i++) d += values[i];
            return d;
        }

        static void print();

        static void delay(double dsec);
};

};

#endif // __TIMER_H__
