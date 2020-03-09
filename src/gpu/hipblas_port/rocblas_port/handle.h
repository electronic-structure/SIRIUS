/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef HANDLE_H
#define HANDLE_H

#include <fstream>
#include <iostream>
#include "rocblas-types.h"
#include "definitions.h"
#include <hip/hip_runtime_api.h>

/*******************************************************************************
 * \brief rocblas_handle is a structure holding the rocblas library context.
 * It must be initialized using rocblas_create_handle() and the returned handle mus
 * It should be destroyed at the end using rocblas_destroy_handle().
 * Exactly like CUBLAS, ROCBLAS only uses one stream for one API routine
******************************************************************************/
namespace {
struct _rocblas_handle
{
    _rocblas_handle();
    ~_rocblas_handle();

    /*******************************************************************************
     * Exactly like CUBLAS, ROCBLAS only uses one stream for one API routine
     ******************************************************************************/

    /*******************************************************************************
     * set stream:
        This API assumes user has already created a valid stream
        Associate the following rocblas API call with this user provided stream
     ******************************************************************************/
    rocblas_status set_stream(hipStream_t user_stream)
    {
        // TODO: check the user_stream valid or not
        rocblas_stream = user_stream;
        return rocblas_status_success;
    }

    /*******************************************************************************
     * get stream
     ******************************************************************************/
    rocblas_status get_stream(hipStream_t* stream) const
    {
        *stream = rocblas_stream;
        return rocblas_status_success;
    }

    // trsm get pointers
    void* get_trsm_Y() const { return trsm_Y; }
    void* get_trsm_invA() const { return trsm_invA; }
    void* get_trsm_invA_C() const { return trsm_invA_C; }

    // trsv get pointers
    void* get_trsv_x() const { return trsv_x; }
    void* get_trsv_alpha() const { return trsv_alpha; }

    rocblas_int device;
    hipDeviceProp_t device_properties;

    // rocblas by default take the system default stream 0 users cannot create
    hipStream_t rocblas_stream = 0;

    // default pointer_mode is on host
    rocblas_pointer_mode pointer_mode = rocblas_pointer_mode_host;

    // space allocated for trsm
    void* trsm_Y      = nullptr;
    void* trsm_invA   = nullptr;
    void* trsm_invA_C = nullptr;

    // space allocated for trsv
    void* trsv_x     = nullptr;
    void* trsv_alpha = nullptr;

    // default logging_mode is no logging
    static rocblas_layer_mode layer_mode;

    // logging streams
    static std::ofstream log_trace_ofs;
    static std::ostream* log_trace_os;
    static std::ofstream log_bench_ofs;
    static std::ostream* log_bench_os;
    static std::ofstream log_profile_ofs;
    static std::ostream* log_profile_os;

    // static data for startup initialization
    static struct init
    {
        init();
    } handle_init;
};


// work buffer size constants
constexpr size_t WORKBUF_TRSM_A_BLKS    = 10;
constexpr size_t WORKBUF_TRSM_B_CHNK    = 32000;
constexpr size_t WORKBUF_TRSM_Y_SZ      = 32000 * 128 * sizeof(double);
constexpr size_t WORKBUF_TRSM_INVA_SZ   = 128 * 128 * 10 * sizeof(double);
constexpr size_t WORKBUF_TRSM_INVA_C_SZ = 128 * 128 * 10 * sizeof(double) / 2;
constexpr size_t WORKBUF_TRSV_X_SZ      = 131072 * sizeof(double);
constexpr size_t WORKBUF_TRSV_ALPHA_SZ  = sizeof(double);

/*******************************************************************************
 * constructor
 ******************************************************************************/
_rocblas_handle::_rocblas_handle()
{
    // default device is active device
    THROW_IF_HIP_ERROR(hipGetDevice(&device));
    THROW_IF_HIP_ERROR(hipGetDeviceProperties(&device_properties, device));

    // rocblas by default take the system default stream 0 users cannot create

    // allocate trsm temp buffers
    THROW_IF_HIP_ERROR(hipMalloc(&trsm_Y, WORKBUF_TRSM_Y_SZ));
    THROW_IF_HIP_ERROR(hipMalloc(&trsm_invA, WORKBUF_TRSM_INVA_SZ));
    THROW_IF_HIP_ERROR(hipMalloc(&trsm_invA_C, WORKBUF_TRSM_INVA_C_SZ));

    // allocate trsv temp buffers
    THROW_IF_HIP_ERROR(hipMalloc(&trsv_x, WORKBUF_TRSV_X_SZ));
    THROW_IF_HIP_ERROR(hipMalloc(&trsv_alpha, WORKBUF_TRSV_ALPHA_SZ));
}

/*******************************************************************************
 * destructor
 ******************************************************************************/
_rocblas_handle::~_rocblas_handle()
{
    if(trsm_Y)
        hipFree(trsm_Y);
    if(trsm_invA)
        hipFree(trsm_invA);
    if(trsm_invA_C)
        hipFree(trsm_invA_C);
    if(trsv_x)
        hipFree(trsv_x);
    if(trsv_alpha)
        hipFree(trsv_alpha);
}

/*******************************************************************************
 * Static handle data
 ******************************************************************************/
rocblas_layer_mode _rocblas_handle::layer_mode = rocblas_layer_mode_none;
std::ofstream _rocblas_handle::log_trace_ofs;
std::ostream* _rocblas_handle::log_trace_os;
std::ofstream _rocblas_handle::log_bench_ofs;
std::ostream* _rocblas_handle::log_bench_os;
std::ofstream _rocblas_handle::log_profile_ofs;
std::ostream* _rocblas_handle::log_profile_os;
_rocblas_handle::init _rocblas_handle::handle_init;

/**
 *  @brief Logging function
 *
 *  @details
 *  open_log_stream Open stream log_os for logging.
 *                  If the environment variable with name environment_variable_name
 *                  is not set, then stream log_os to std::cerr.
 *                  Else open a file at the full logfile path contained in
 *                  the environment variable.
 *                  If opening the file suceeds, stream to the file
 *                  else stream to std::cerr.
 *
 *  @param[in]
 *  environment_variable_name   const char*
 *                              Name of environment variable that contains
 *                              the full logfile path.
 *
 *  @parm[out]
 *  log_os      std::ostream*&
 *              Output stream. Stream to std:cerr if environment_variable_name
 *              is not set, else set to stream to log_ofs
 *
 *  @parm[out]
 *  log_ofs     std::ofstream&
 *              Output file stream. If log_ofs->is_open()==true, then log_os
 *              will stream to log_ofs. Else it will stream to std::cerr.
 */

static void open_log_stream(const char* environment_variable_name,
                            std::ostream*& log_os,
                            std::ofstream& log_ofs)

{
    // By default, output to cerr
    log_os = &std::cerr;

    // if environment variable is set, open file at logfile_pathname contained in the
    // environment variable
    auto logfile_pathname = getenv(environment_variable_name);
    if(logfile_pathname)
    {
        log_ofs.open(logfile_pathname, std::ios_base::trunc);

        // if log_ofs is open, then stream to log_ofs, else log_os is already set to std::cerr
        if(log_ofs.is_open())
            log_os = &log_ofs;
    }
}

/*******************************************************************************
 * Static runtime initialization
 ******************************************************************************/
_rocblas_handle::init::init()
{
    // set layer_mode from value of environment variable ROCBLAS_LAYER
    auto str_layer_mode = getenv("ROCBLAS_LAYER");
    if(str_layer_mode)
    {
        layer_mode = static_cast<rocblas_layer_mode>(strtol(str_layer_mode, 0, 0));

        // open log_trace file
        if(layer_mode & rocblas_layer_mode_log_trace)
            open_log_stream("ROCBLAS_LOG_TRACE_PATH", log_trace_os, log_trace_ofs);

        // open log_bench file
        if(layer_mode & rocblas_layer_mode_log_bench)
            open_log_stream("ROCBLAS_LOG_BENCH_PATH", log_bench_os, log_bench_ofs);

        // open log_profile file
        if(layer_mode & rocblas_layer_mode_log_profile)
            open_log_stream("ROCBLAS_LOG_PROFILE_PATH", log_profile_os, log_profile_ofs);
    }
}


}

#endif
