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

#include "rocblas_port.h"
#include <hipblas.h>
#include "rocblas_port/port_hip_roc_translation.h"

hipblasStatus_t hipblas_port_Sgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const float* alpha,
                                   const float* A, int lda, const float* x, int incx, const float* beta, float* y,
                                   int incy)
{
    return rocBLASStatusToHIPStatus(rocblas_port_sgemv((rocblas_handle)handle, hipOperationToHCCOperation(trans), m, n,
                                                       alpha, A, lda, x, incx, beta, y, incy));
}

hipblasStatus_t hipblas_port_Dgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const double* alpha,
                                   const double* A, int lda, const double* x, int incx, const double* beta, double* y,
                                   int incy)
{
    return rocBLASStatusToHIPStatus(rocblas_port_dgemv((rocblas_handle)handle, hipOperationToHCCOperation(trans), m, n,
                                                       alpha, A, lda, x, incx, beta, y, incy));
}

hipblasStatus_t hipblas_port_Cgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n,
                                   const hipFloatComplex* alpha, const hipFloatComplex* A, int lda,
                                   const hipFloatComplex* x, int incx, const hipFloatComplex* beta, hipFloatComplex* y,
                                   int incy)
{
    return rocBLASStatusToHIPStatus(rocblas_port_cgemv((rocblas_handle)handle, hipOperationToHCCOperation(trans), m, n,
                                                       alpha, A, lda, x, incx, beta, y, incy));
}

hipblasStatus_t hipblas_port_Zgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n,
                                   const hipDoubleComplex* alpha, const hipDoubleComplex* A, int lda,
                                   const hipDoubleComplex* x, int incx, const hipDoubleComplex* beta,
                                   hipDoubleComplex* y, int incy)
{
    return rocBLASStatusToHIPStatus(rocblas_port_zgemv((rocblas_handle)handle, hipOperationToHCCOperation(trans), m, n,
                                                       alpha, A, lda, x, incx, beta, y, incy));
}

hipblasStatus_t hipblas_port_Sgemm(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m,
                                   int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb,
                                   const float* beta, float* C, int ldc)
{
    return rocBLASStatusToHIPStatus(rocblas_port_sgemm((rocblas_handle)handle, hipOperationToHCCOperation(transa),
                                                       hipOperationToHCCOperation(transb), m, n, k, alpha, A, lda, B,
                                                       ldb, beta, C, ldc));
}

hipblasStatus_t hipblas_port_Dgemm(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m,
                                   int n, int k, const double* alpha, const double* A, int lda, const double* B,
                                   int ldb, const double* beta, double* C, int ldc)
{
    return rocBLASStatusToHIPStatus(rocblas_port_dgemm((rocblas_handle)handle, hipOperationToHCCOperation(transa),
                                                       hipOperationToHCCOperation(transb), m, n, k, alpha, A, lda, B,
                                                       ldb, beta, C, ldc));
}

hipblasStatus_t hipblas_port_Cgemm(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m,
                                   int n, int k, const hipFloatComplex* alpha, const hipFloatComplex* A, int lda,
                                   const hipFloatComplex* B, int ldb, const hipFloatComplex* beta, hipFloatComplex* C,
                                   int ldc)
{
    return rocBLASStatusToHIPStatus(rocblas_port_cgemm((rocblas_handle)handle, hipOperationToHCCOperation(transa),
                                                       hipOperationToHCCOperation(transb), m, n, k, alpha, A, lda, B,
                                                       ldb, beta, C, ldc));
}

hipblasStatus_t hipblas_port_Zgemm(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m,
                                   int n, int k, const hipDoubleComplex* alpha, const hipDoubleComplex* A, int lda,
                                   const hipDoubleComplex* B, int ldb, const hipDoubleComplex* beta,
                                   hipDoubleComplex* C, int ldc)
{
    return rocBLASStatusToHIPStatus(rocblas_port_zgemm((rocblas_handle)handle, hipOperationToHCCOperation(transa),
                                                       hipOperationToHCCOperation(transb), m, n, k, alpha, A, lda, B,
                                                       ldb, beta, C, ldc));
}

hipblasStatus_t hipblas_port_Strmm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo,
                                   hipblasOperation_t trans, hipblasDiagType_t diag, int m, int n, const float* alpha,
                                   const float* A, int lda, const float* B, int ldb, float* C, int ldc)
{

    return rocBLASStatusToHIPStatus(rocblas_port_strmm(
        (rocblas_handle)handle, hipSideToHCCSide(side), hipFillToHCCFill(uplo), hipOperationToHCCOperation(trans),
        hipDiagonalToHCCDiagonal(diag), m, n, alpha, A, lda, B, ldb, C, ldc));
}

hipblasStatus_t hipblas_port_Dtrmm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo,
                                   hipblasOperation_t trans, hipblasDiagType_t diag, int m, int n, const double* alpha,
                                   const double* A, int lda, const double* B, int ldb, double* C, int ldc)
{
    return rocBLASStatusToHIPStatus(rocblas_port_dtrmm(
        (rocblas_handle)handle, hipSideToHCCSide(side), hipFillToHCCFill(uplo), hipOperationToHCCOperation(trans),
        hipDiagonalToHCCDiagonal(diag), m, n, alpha, A, lda, B, ldb, C, ldc));
}

hipblasStatus_t hipblas_port_Ctrmm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo,
                                   hipblasOperation_t trans, hipblasDiagType_t diag, int m, int n,
                                   const hipFloatComplex* alpha, const hipFloatComplex* A, int lda,
                                   const hipFloatComplex* B, int ldb, hipFloatComplex* C, int ldc)
{
    return rocBLASStatusToHIPStatus(rocblas_port_ctrmm(
        (rocblas_handle)handle, hipSideToHCCSide(side), hipFillToHCCFill(uplo), hipOperationToHCCOperation(trans),
        hipDiagonalToHCCDiagonal(diag), m, n, alpha, A, lda, B, ldb, C, ldc));
}

hipblasStatus_t hipblas_port_Ztrmm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo,
                                   hipblasOperation_t trans, hipblasDiagType_t diag, int m, int n,
                                   const hipDoubleComplex* alpha, const hipDoubleComplex* A, int lda,
                                   const hipDoubleComplex* B, int ldb, hipDoubleComplex* C, int ldc)
{
    return rocBLASStatusToHIPStatus(rocblas_port_ztrmm(
        (rocblas_handle)handle, hipSideToHCCSide(side), hipFillToHCCFill(uplo), hipOperationToHCCOperation(trans),
        hipDiagonalToHCCDiagonal(diag), m, n, alpha, A, lda, B, ldb, C, ldc));
}



/*
 * GER
 */
hipblasStatus_t hipblas_port_Sger(hipblasHandle_t handle, int m, int n, const float* alpha, const float* x, int incx,
                                  const float* y, int incy, float* A, int lda)
{
    return rocBLASStatusToHIPStatus(rocblas_port_sger((rocblas_handle)handle, m, n, alpha, x, incx, y, incy, A, lda));
}

hipblasStatus_t hipblas_port_Dger(hipblasHandle_t handle, int m, int n, const double* alpha, const double* x, int incx,
                                  const double* y, int incy, double* A, int lda)
{
    return rocBLASStatusToHIPStatus(rocblas_port_dger((rocblas_handle)handle, m, n, alpha, x, incx, y, incy, A, lda));
}

hipblasStatus_t hipblas_port_Cgeru(hipblasHandle_t handle, int m, int n, const hipFloatComplex* alpha,
                                   const hipFloatComplex* x, int incx, const hipFloatComplex* y, int incy,
                                   hipFloatComplex* A, int lda)
{
    return rocBLASStatusToHIPStatus(rocblas_port_cgeru((rocblas_handle)handle, m, n, alpha, x, incx, y, incy, A, lda));
}

hipblasStatus_t hipblas_port_Zgeru(hipblasHandle_t handle, int m, int n, const hipDoubleComplex* alpha,
                                   const hipDoubleComplex* x, int incx, const hipDoubleComplex* y, int incy,
                                   hipDoubleComplex* A, int lda)
{
    return rocBLASStatusToHIPStatus(rocblas_port_zgeru((rocblas_handle)handle, m, n, alpha, x, incx, y, incy, A, lda));
}

/*
 * AXPY
 */
hipblasStatus_t hipblas_port_Saxpy(hipblasHandle_t handle, int n, const float* alpha, const float* x, int incx,
                                   float* y, int incy)
{
    return rocBLASStatusToHIPStatus(rocblas_port_saxpy((rocblas_handle)handle, n, alpha, x, incx, y, incy));
}

hipblasStatus_t hipblas_port_Daxpy(hipblasHandle_t handle, int n, const double* alpha, const double* x, int incx,
                                   double* y, int incy)
{
    return rocBLASStatusToHIPStatus(rocblas_port_daxpy((rocblas_handle)handle, n, alpha, x, incx, y, incy));
}

hipblasStatus_t hipblas_port_Caxpy(hipblasHandle_t handle, int n, const hipFloatComplex* alpha,
                                   const hipFloatComplex* x, int incx, hipFloatComplex* y, int incy)
{
    return rocBLASStatusToHIPStatus(rocblas_port_caxpy((rocblas_handle)handle, n, alpha, x, incx, y, incy));
}

hipblasStatus_t hipblas_port_Zaxpy(hipblasHandle_t handle, int n, const hipDoubleComplex* alpha,
                                   const hipDoubleComplex* x, int incx, hipDoubleComplex* y, int incy)
{
    return rocBLASStatusToHIPStatus(rocblas_port_zaxpy((rocblas_handle)handle, n, alpha, x, incx, y, incy));
}
