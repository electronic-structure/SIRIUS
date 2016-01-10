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

/** \file sirius_internal.h
 *   
 *  \brief Contains basic definitions and declarations.
 */

#ifndef __SIRIUS_INTERNAL_H__
#define __SIRIUS_INTERNAL_H__

#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <gsl/gsl_sf_bessel.h>
#include <fftw3.h>
#include <vector>
#include <complex>
#include <iostream>
#include <algorithm>
#include "communicator.h"
#include "gpu.h"

/// Namespace of the SIRIUS library.
namespace sirius {

    inline void initialize(bool call_mpi_init__)
    {
        if (call_mpi_init__) Communicator::initialize();

        #ifdef __GPU
        //cuda_initialize();
        #if defined(__VERBOSITY) && (__VERBOSITY > 0)
        if (mpi_comm_world().rank() == 0) cuda_device_info();
        #endif
        cuda_create_streams(omp_get_max_threads() + 1);
        cublas_create_handles(omp_get_max_threads() + 1);
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

        assert(sizeof(int) == 4);
        assert(sizeof(double) == 8);
    }

    inline void finalize()
    {
        Communicator::finalize();
        #ifdef __MAGMA
        magma_finalize_wrapper();
        #endif
        #ifdef __LIBSCI_ACC
        libsci_acc_finalize();
        #endif
        #ifdef __GPU
        cublas_destroy_handles(omp_get_max_threads() + 1);
        cuda_destroy_streams();
        cuda_device_reset();
        #endif
        fftw_cleanup();
    }

};

#endif // __SIRIUS_INTERNAL_H__

