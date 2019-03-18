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

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include "rocblas_port/rocblas-types.h"
#include "rocblas_port/status.h"
#include "rocblas_port/definitions.h"
#include "rocblas_port/handle.h"
#include "rocblas_port/utility.h"
#include "rocblas_port/reduction.h"
#include "rocblas_port/port_helper_func.h"

namespace {

template <typename U, typename T>
__global__ void rocblas_axpy_kernel(const rocblas_int n, U alpha_device_host, const T* __restrict__ x,
                                   const rocblas_int incx, T* y, const rocblas_int incy)
{
    auto alpha = load_scalar(alpha_device_host);

    auto row = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    if (row < n)
        y[row * incy] += alpha * x[row * incx];
}

template <typename T>
rocblas_status rocblas_axpy(rocblas_handle handle, rocblas_int n, const T* alpha, const T* x, rocblas_int incx, T* y,
                            rocblas_int incy)
{
    if (!handle)
        return rocblas_status_invalid_handle;
    if (!alpha)
        return rocblas_status_invalid_pointer;
    if (!n)
        return rocblas_status_success;
    if (!x || !y)
        return rocblas_status_invalid_pointer;

    dim3 threads(256);
    dim3 grid(n / 256 + (n % 256 != 0));
    hipStream_t rocblas_stream = handle->rocblas_stream;

    if (handle->pointer_mode == rocblas_pointer_mode_device) {
        hipLaunchKernelGGL((rocblas_axpy_kernel), grid, threads, 0, rocblas_stream, n, alpha, x, incx, y, incy);
    } else {
        hipLaunchKernelGGL((rocblas_axpy_kernel), grid, threads, 0, rocblas_stream, n, *alpha, x, incx, y, incy);
    }
    return rocblas_status_success;
}

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_port_saxpy(rocblas_handle handle, rocblas_int n, const float* alpha, const float* x,
                                  rocblas_int incx, float* y, rocblas_int incy)
{
    return rocblas_axpy(handle, n, alpha, x, incx, y, incy);
}

rocblas_status rocblas_port_daxpy(rocblas_handle handle, rocblas_int n, const double* alpha, const double* x,
                                  rocblas_int incx, double* y, rocblas_int incy)
{
    return rocblas_axpy(handle, n, alpha, x, incx, y, incy);
}

rocblas_status rocblas_port_caxpy(rocblas_handle handle, rocblas_int n, const hipFloatComplex* alpha,
                                  const hipFloatComplex* x, rocblas_int incx, hipFloatComplex* y, rocblas_int incy)
{
    return rocblas_axpy(handle, n, alpha, x, incx, y, incy);
}
rocblas_status rocblas_port_zaxpy(rocblas_handle handle, rocblas_int n, const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* x, rocblas_int incx, hipDoubleComplex* y, rocblas_int incy)
{
    return rocblas_axpy(handle, n, alpha, x, incx, y, incy);
}

} // extern "C"
