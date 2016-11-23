// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

/** \file config.h
 *
 *  \brief Main configuration header.
 */

#ifndef __CONFIG_H__
#define __CONFIG_H__

#define __TIMER
//== #define __TIMER_TIMEOFDAY
//== #define __TIMER_MPI_WTIME
#define __TIMER_CHRONO

//== #define __PRINT_OBJECT_HASH

//== #define __PRINT_OBJECT_CHECKSUM

//== #define __PRINT_MEMORY_USAGE

//== #define __SCALAPACK

//== #define __PILAENV_BLOCKSIZE=2048

//== #define __ELPA

//== #define __PLASMA

//== #define __MAGMA

//== #definr __LIBSCI_ACC

#ifdef __LIBSCI_ACC
#warning "Don't forget to use pinned memory with libsci_acc"
#endif

//== #define __GPU

//== #define __GPU_DIRECT

/* link with -lnvToolsExt if you use NVIDIA Tools Extension Library (NVTX) */
//== #define __GPU_NVTX

//== #define __RS_GEN_EIG

//== #if !defined(NDEBUG)
//== #pragma message("NDEBUG is not defined. Assert statements are enabled.")
//== #endif

#define __PROFILE
//#define __PROFILE_STACK
#define __PROFILE_TIME
//#define __PROFILE_FUNC

#if defined(__LIBSCI_ACC) && !defined(__GPU)
#error "GPU interface must be enabled for libsci_acc"
#endif

const bool test_spinor_wf = false;

const bool hdf5_trace_errors = false;

const bool check_pseudo_charge = false;

/// Level of internal verification
/** __VERIFICATION = 0 : nothing to do \n
 *  __VERIFICATION = 1 : basic checkes \n */
#ifndef __VERIFICATION
#define __VERIFICATION 0
#endif

const bool use_second_variation = true;

#endif // __CONFIG_H__
