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
#ifdef __TIMER_CHRONO
#include <chrono>
#endif

namespace sirius 
{

struct timer_stats
{
    int count;
    double min_value;
    double max_value;
    double total_value;
    double average_value;
};

/// Simple timer interface.
class Timer
{
    private:
        
        #ifdef __TIMER
        /// String label of the timer.
        std::string label_;
        
        /// Starting time.
        #if defined(__TIMER_TIMEOFDAY)
        timeval starting_time_;
        #elif defined(__TIMER_MPI_WTIME)
        double starting_time_;
        #elif defined(__TIMER_CHRONO)
        std::chrono::high_resolution_clock::time_point starting_time_;
        #endif

        Communicator const* comm_;

        /// true if timer is running
        bool active_;

        #endif
    
    public:
        
        #ifdef __TIMER
        Timer(std::string const& label__) 
            : label_(label__),
              comm_(nullptr),
              active_(false)
        {
            if (timers().count(label_) == 0) timers()[label_] = std::vector<double>();

            start();
        }
        #else
        Timer(std::string const& label__)
        {
        }
        #endif
        
        #ifdef __TIMER
        Timer(std::string const& label__, Communicator const& comm__) 
            : label_(label__),
              comm_(&comm__),
              active_(false)
        {
            if (timers().count(label_) == 0) timers()[label_] = std::vector<double>();

            start();
        }
        #else
        Timer(std::string const& label__, Communicator const& comm__)
        {
        }
        #endif

        ~Timer()
        {
            #ifdef __TIMER
            if (active_) stop();
            #endif
        }

        void start();

        double stop();

        //double value();

        static void clear()
        {
            #ifdef __TIMER
            timers().clear();
            #endif
        }
        
        #ifdef __TIMER
        static std::map< std::string, std::vector<double> >& timers()
        {
            static std::map< std::string, std::vector<double> > timers_;
            return timers_;
        }

        static std::map< std::string, timer_stats> collect_timer_stats();

        static double value(std::string const& label__)
        {
            auto values = timers()[label__];

            double d = 0;
            for (double v: values) d += v;
            return d;
        }

        static void print();
        #endif
};

};

#endif // __TIMER_H__
