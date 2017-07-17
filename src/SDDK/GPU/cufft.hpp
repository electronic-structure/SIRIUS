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

/** \file cufft.hpp
 *   
 *  \brief Interface to cuFFT related functions.
 */
#ifndef __CUFFT_HPP__
#define __CUFFT_HPP__

#include <unistd.h>
#include <cufft.h>
#include <cuda.h>
#include "cuda.hpp"

namespace cufft {

inline void error_message(cufftResult result)
{
    switch (result) {
        case CUFFT_INVALID_PLAN: {
            printf("CUFFT_INVALID_PLAN: the plan parameter is not a valid handle\n");
            break;
        }
        case CUFFT_ALLOC_FAILED: {
            printf("CUFFT_ALLOC_FAILED: cuFFT failed to allocate GPU or CPU memory\n");
            break;
        }
        case CUFFT_INVALID_VALUE: {
            printf("CUFFT_INVALID_VALUE: at least one of the parameters idata, odata, and direction is not valid\n");
            break;
        }
        case CUFFT_INTERNAL_ERROR: {
            printf("CUFFT_INTERNAL_ERROR: an internal driver error was detected\n");
            break;
        }
        case CUFFT_SETUP_FAILED: {
            printf("CUFFT_SETUP_FAILED: the cuFFT library failed to initialize\n");
            break;
        }
        case CUFFT_INVALID_SIZE: {
            printf("CUFFT_INVALID_SIZE: user specified an invalid transform size\n");
            break;
        }
        case CUFFT_EXEC_FAILED: {
            printf("CUFFT_EXEC_FAILED: cuFFT failed to execute the transform on the GPU\n");
            break;
        }
        default: {
            printf("unknown error code %i\n", result);
            break;
        }
    }
}

#define CALL_CUFFT(func__, args__)                                                  \
{                                                                                   \
    cufftResult result;                                                             \
    if ((result = func__ args__) != CUFFT_SUCCESS) {                                \
        char nm[1024];                                                              \
        gethostname(nm, 1024);                                                      \
        printf("hostname: %s\n", nm);                                               \
        printf("Error in %s at line %i of file %s: ", #func__, __LINE__, __FILE__); \
        cufft::error_message(result);                                                      \
        exit(-100);                                                                 \
    }                                                                               \
}

inline void create_plan_handle(cufftHandle* plan)
{
    CALL_CUFFT(cufftCreate, (plan));
}

inline void destroy_plan_handle(cufftHandle plan)
{
    CALL_CUFFT(cufftDestroy, (plan));
}

// Size of work buffer in bytes
inline size_t get_work_size(int ndim, int* dims, int nfft)
{
    int fft_size = 1;
    for (int i = 0; i < ndim; i++) {
        fft_size *= dims[i];
    }
    size_t work_size;

    CALL_CUFFT(cufftEstimateMany, (ndim, dims, NULL, 1, fft_size, NULL, 1, fft_size, CUFFT_Z2Z, nfft, &work_size));
    
    return work_size;
}

inline size_t create_batch_plan(cufftHandle plan, int rank, int* dims, int* embed, int stride, int dist, int nfft, int auto_alloc)
{
    int fft_size = 1;
    for (int i = 0; i < rank; i++) {
        fft_size *= dims[i];
    }
    
    if (auto_alloc) {
        CALL_CUFFT(cufftSetAutoAllocation, (plan, true));
    } else {
        CALL_CUFFT(cufftSetAutoAllocation, (plan, false));
    }
    size_t work_size;

    /* 1D
       input[ b * idist + x * istride]
       output[ b * odist + x * ostride]
       
       2D
       input[b * idist + (x * inembed[1] + y) * istride]
       output[b * odist + (x * onembed[1] + y) * ostride]
       
       3D
       input[b * idist + ((x * inembed[1] + y) * inembed[2] + z) * istride]
       output[b * odist + ((x * onembed[1] + y) * onembed[2] + z) * ostride]

       - See more at: http://docs.nvidia.com/cuda/cufft/index.html#advanced-data-layout
     */
    CALL_CUFFT(cufftMakePlanMany, (plan, rank, dims, embed, stride, dist, embed, stride, dist, CUFFT_Z2Z, nfft, &work_size));

    return work_size;
}

inline void set_work_area(cufftHandle plan, void* work_area)
{
    CALL_CUFFT(cufftSetWorkArea, (plan, work_area));
}

inline void set_stream(cufftHandle plan__, int stream_id__)
{
    CALL_CUFFT(cufftSetStream, (plan__, acc::stream(stream_id__)));
}

inline void forward_transform(cufftHandle plan, cuDoubleComplex* fft_buffer)
{
    //CUDA_timer t("cufft_forward_transform");
    CALL_CUFFT(cufftExecZ2Z, (plan, fft_buffer, fft_buffer, CUFFT_FORWARD));
}

inline void backward_transform(cufftHandle plan, cuDoubleComplex* fft_buffer)
{
    //CUDA_timer t("cufft_backward_transform");
    CALL_CUFFT(cufftExecZ2Z, (plan, fft_buffer, fft_buffer, CUFFT_INVERSE));
}

} // namespace cufft
#endif
