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

/** \file rocfft_interace.hpp
 *   
 *  \brief Interface to rocFFT related functions.
 */

#ifndef __ROCFFT_INTERFACE_HPP__
#define __ROCFFT_INTERFACE_HPP__
#include<complex>
#include "acc.hpp"

namespace rocfft
{
void destroy_plan_handle(void* plan);

// NOTE: creates a new plan for work size calculation; if a plan is available call directly with
// pointer to it for better performance
size_t get_work_size(int ndim, int* dims, int nfft);

size_t get_work_size(void* plan);

// embed can be nullptr (stride and dist are then ignored)
void* create_batch_plan(int rank, int* dims, int* embed, int stride, int dist, int nfft,
                        bool auto_alloc);

void set_work_area(void* plan, void* work_area);

void set_stream(void* plan__, stream_id sid__);

void forward_transform(void* plan, std::complex<double>* fft_buffer);

void backward_transform(void* plan, std::complex<double>* fft_buffer);


// function for rocfft library initializeation
// NOTE: Source code in ROCM suggests nothing is actually done (empty functions)
void initialize();

void finalize();

} // namespace rocfft
#endif
