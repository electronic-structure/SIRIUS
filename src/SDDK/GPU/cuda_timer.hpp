
// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file cuda_timer.hpp
 *
 *  \brief Timer for CUDA kernels.
 */

#ifndef __CUDA_TIMER_HPP__
#define __CUDA_TIMER_HPP__
#include <execinfo.h>
#include <signal.h>
#include <assert.h>
#include <map>
#include <vector>
#include <string>
#include <stdio.h>

class CUDA_timers_wrapper
{
    private:

        std::map<std::string, std::vector<float> > cuda_timers_;

    public:

        void add_measurment(std::string const& label, float value)
        {
            cuda_timers_[label].push_back(value / 1000);
        }

        void print()
        {
            std::printf("\n");
            std::printf("CUDA timers \n");
            for (int i = 0; i < 115; i++) std::printf("-");
            std::printf("\n");
            std::printf("name                                                              count      total        min        max    average\n");
            for (int i = 0; i < 115; i++) std::printf("-");
            std::printf("\n");

            std::map<std::string, std::vector<float> >::iterator it;
            for (it = cuda_timers_.begin(); it != cuda_timers_.end(); it++) {
                int count = (int)it->second.size();
                double total = 0.0;
                float minval = 1e10;
                float maxval = 0.0;
                for (int i = 0; i < count; i++) {
                    total += it->second[i];
                    minval = std::min(minval, it->second[i]);
                    maxval = std::max(maxval, it->second[i]);
                }
                double average = (count == 0) ? 0.0 : total / count;
                if (count == 0) {
                    minval = 0.0;
                }

                std::printf("%-60s :    %5i %10.4f %10.4f %10.4f %10.4f\n", it->first.c_str(), count, total, minval, maxval, average);
            }
        }
};

class CUDA_timer
{
    private:

        cudaEvent_t e_start_;
        cudaEvent_t e_stop_;
        bool active_;
        std::string label_;

        void start()
        {
            cudaEventCreate(&e_start_);
            cudaEventCreate(&e_stop_);
            cudaEventRecord(e_start_, 0);
        }

        void stop()
        {
            float time;
            cudaEventRecord(e_stop_, 0);
            cudaEventSynchronize(e_stop_);
            cudaEventElapsedTime(&time, e_start_, e_stop_);
            cudaEventDestroy(e_start_);
            cudaEventDestroy(e_stop_);
            cuda_timers_wrapper().add_measurment(label_, time);
            active_ = false;
        }

    public:

        CUDA_timer(std::string const& label__) : label_(label__), active_(false)
        {
            start();
        }

        ~CUDA_timer()
        {
            stop();
        }

        static CUDA_timers_wrapper& cuda_timers_wrapper()
        {
            static CUDA_timers_wrapper cuda_timers_wrapper_;
            return cuda_timers_wrapper_;
        }
};

#endif
