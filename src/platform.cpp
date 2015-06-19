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

/** \file platform.cpp
 *   
 *  \brief Contains remaining implementation of Platform class.
 */

#include <fftw3-mpi.h>
#include <fftw3.h>
#include "platform.h"

int Platform::num_fft_threads_ = -1;

#ifdef __PLASMA
extern "C" void plasma_init(int num_cores);
#endif

#ifdef __LIBSCI_ACC
extern "C" void libsci_acc_init();
extern "C" void libsci_acc_finalize();
#endif

void Platform::initialize(bool call_mpi_init__)
{
    //if (call_mpi_init__) MPI_Init(NULL, NULL);

    int provided;
    if (call_mpi_init__) 
    {
        MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
    }

    MPI_Query_thread(&provided);
    if (provided != MPI_THREAD_FUNNELED && rank() == 0)
    {
        printf("Warning! MPI_THREAD_FUNNELED level of thread support is not provided.\n");
    }

    #ifdef __GPU
    //cuda_initialize();
    #if defined(__VERBOSITY) && (__VERBOSITY > 0)
    if (rank() == 0) cuda_device_info();
    #endif
    cuda_create_streams(max_num_threads() + 1);
    cublas_create_handles(max_num_threads() + 1);
    #endif
    #ifdef __MAGMA
    magma_init_wrapper();
    #endif
    #ifdef __PLASMA
    plasma_init(max_num_threads());
    #endif
    #ifdef __LIBSCI_ACC
    libsci_acc_init();
    #endif
    #ifdef __FFTW_THREADED
    if (provided >= MPI_THREAD_FUNNELED)
    {
        if (!fftw_init_threads())
        {
            printf("error in fftw_init_threads()\n");
            exit(0);
        }
    }
    #endif
    fftw_mpi_init();

    assert(sizeof(int) == 4);
    assert(sizeof(double) == 8);
}

void Platform::finalize()
{
    MPI_Finalize();
    #ifdef __MAGMA
    magma_finalize_wrapper();
    #endif
    #ifdef __LIBSCI_ACC
    libsci_acc_finalize();
    #endif
    #ifdef __GPU
    cublas_destroy_handles(max_num_threads() + 1);
    cuda_destroy_streams(max_num_threads() + 1);
    cuda_device_reset();
    #endif
    fftw_mpi_cleanup();
    #ifdef __FFTW_THREADED
    fftw_cleanup_threads();
    #endif
    fftw_cleanup();
}

void Platform::abort()
{
    MPI_Abort(MPI_COMM_WORLD, -13);
}
