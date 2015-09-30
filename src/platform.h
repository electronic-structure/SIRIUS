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

/** \file platform.h
 *   
 *  \brief Contains definition and implementation of Platform class.
 */

#ifndef __PLATFORM_H__
#define __PLATFORM_H__

#include <mpi.h>
#include <omp.h>
#include <signal.h>
#include <unistd.h>
#include <vector>
#include <fstream>
#include "config.h"
#ifdef __GPU
#include "gpu.h"
#endif
#include "typedefs.h"

/// Platform specific functions.
class Platform
{
    private:

        static int num_fft_threads_;
    
    public:

        static void initialize(bool call_mpi_init);

        static void finalize();

        static void abort();

        static int rank() // TODO: global rank should never be referenced; all calculations "live" inside thier own communicator
        {
            int r;
            MPI_Comm_rank(MPI_COMM_WORLD, &r);
            return r;
        }

        /// Returm maximum number of OMP threads.
        /** Maximum number of OMP threads is controlled by environment variable OMP_NUM_THREADS */
        static inline int max_num_threads()
        {
            return omp_get_max_threads();
        }

        /// Returm number of actually running OMP threads. 
        static inline int num_threads()
        {
            return omp_get_num_threads();
        }
        
        /// Return thread id.
        static inline int thread_id()
        {
            return omp_get_thread_num();
        }
};

#endif
