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

template <typename T, typename U>
__global__ void rocblas_ger_kernel(const rocblas_int rows_A, const rocblas_int cols_A, const T* __restrict__ x,
                                   const rocblas_int incx, const T* __restrict__ y, rocblas_int incy,
                                   U alpha_device_host, T* A, const rocblas_int lda)
{
    auto alpha = load_scalar(alpha_device_host);

    auto row = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    auto col = hipBlockIdx_y;

    if (row < rows_A)
        A[row + col * lda] += alpha * x[row * incx] * y[col * incy];
}

template <typename T>
rocblas_status rocblas_ger(rocblas_handle handle, rocblas_int m, rocblas_int n, const T* alpha, const T* x, rocblas_int incx,
            const T* y, rocblas_int incy, T* A, rocblas_int lda)
{
    if (!handle)
        return rocblas_status_invalid_handle;
    if (!alpha)
        return rocblas_status_invalid_pointer;
    if (!n || !m)
        return rocblas_status_success;
    if (!A || !x || !y)
        return rocblas_status_invalid_pointer;

    dim3 threads(256);
    dim3 grid(m / 256 + (m % 256 != 0), n);
    hipStream_t rocblas_stream = handle->rocblas_stream;

    if (handle->pointer_mode == rocblas_pointer_mode_device) {
            hipLaunchKernelGGL((rocblas_ger_kernel), grid, threads, 0, rocblas_stream,
                               m, n, x, incx, y, incy, alpha, A, lda);
    } else {
            hipLaunchKernelGGL((rocblas_ger_kernel), grid, threads, 0, rocblas_stream,
                               m, n, x, incx, y, incy, *alpha, A, lda);
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

rocblas_status rocblas_port_sger(rocblas_handle handle, rocblas_int m, rocblas_int n, const float* alpha,
                                 const float* x, rocblas_int incx, const float* y, rocblas_int incy, float* A,
                                 rocblas_int lda)
{
    return rocblas_ger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

rocblas_status rocblas_port_dger(rocblas_handle handle, rocblas_int m, rocblas_int n, const double* alpha,
                                 const double* x, rocblas_int incx, const double* y, rocblas_int incy, double* A,
                                 rocblas_int lda)
{
    return rocblas_ger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

rocblas_status rocblas_port_cgeru(rocblas_handle handle, rocblas_int m, rocblas_int n, const hipFloatComplex* alpha,
                                  const hipFloatComplex* x, rocblas_int incx, const hipFloatComplex* y,
                                  rocblas_int incy, hipFloatComplex* A, rocblas_int lda)
{
    return rocblas_ger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}
rocblas_status rocblas_port_zgeru(rocblas_handle handle, rocblas_int m, rocblas_int n, const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* x, rocblas_int incx, const hipDoubleComplex* y,
                                  rocblas_int incy, hipDoubleComplex* A, rocblas_int lda)
{
    return rocblas_ger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

} // extern "C"
