#ifndef _ROCBLAS_PORT_H_
#define _ROCBLAS_PORT_H_

#include <hip/hip_complex.h>
#include "rocblas_port/rocblas-types.h"
#include "rocblas_port/handle.h"

extern "C" {

/*
 * GEMV
 */
rocblas_status rocblas_port_sgemv(rocblas_handle handle, rocblas_operation transA, rocblas_int m, rocblas_int n,
                                  const float* alpha, const float* A, rocblas_int lda, const float* x, rocblas_int incx,
                                  const float* beta, float* y, rocblas_int incy);

rocblas_status rocblas_port_dgemv(rocblas_handle handle, rocblas_operation transA, rocblas_int m, rocblas_int n,
                                  const double* alpha, const double* A, rocblas_int lda, const double* x,
                                  rocblas_int incx, const double* beta, double* y, rocblas_int incy);

rocblas_status rocblas_port_cgemv(rocblas_handle handle, rocblas_operation transA, rocblas_int m, rocblas_int n,
                                  const hipFloatComplex* alpha, const hipFloatComplex* A, rocblas_int lda,
                                  const hipFloatComplex* x, rocblas_int incx, const hipFloatComplex* beta,
                                  hipFloatComplex* y, rocblas_int incy);

rocblas_status rocblas_port_zgemv(rocblas_handle handle, rocblas_operation transA, rocblas_int m, rocblas_int n,
                                  const hipDoubleComplex* alpha, const hipDoubleComplex* A, rocblas_int lda,
                                  const hipDoubleComplex* x, rocblas_int incx, const hipDoubleComplex* beta,
                                  hipDoubleComplex* y, rocblas_int incy);

/*
 * GEMM
 */
rocblas_status rocblas_port_sgemm(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb,
                                  rocblas_int m, rocblas_int n, rocblas_int k, const float* alpha, const float* A,
                                  rocblas_int lda, const float* B, rocblas_int ldb, const float* beta, float* C,
                                  rocblas_int ldc);

rocblas_status rocblas_port_dgemm(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb,
                                  rocblas_int m, rocblas_int n, rocblas_int k, const double* alpha, const double* A,
                                  rocblas_int lda, const double* B, rocblas_int ldb, const double* beta, double* C,
                                  rocblas_int ldc);

rocblas_status rocblas_port_cgemm(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb,
                                  rocblas_int m, rocblas_int n, rocblas_int k, const hipFloatComplex* alpha,
                                  const hipFloatComplex* A, rocblas_int lda, const hipFloatComplex* B, rocblas_int ldb,
                                  const hipFloatComplex* beta, hipFloatComplex* C, rocblas_int ldc);

rocblas_status rocblas_port_zgemm(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb,
                                  rocblas_int m, rocblas_int n, rocblas_int k, const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* A, rocblas_int lda, const hipDoubleComplex* B,
                                  rocblas_int ldb, const hipDoubleComplex* beta, hipDoubleComplex* C, rocblas_int ldc);

/*
 * TRMM
 */

rocblas_status rocblas_port_strmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
                                  rocblas_diagonal diag, rocblas_int m, rocblas_int n, const float* alpha,
                                  const float* A, rocblas_int lda, const float* B, rocblas_int ldb, float* C,
                                  rocblas_int ldc);

rocblas_status rocblas_port_dtrmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
                                  rocblas_diagonal diag, rocblas_int m, rocblas_int n, const double* alpha,
                                  const double* A, rocblas_int lda, const double* B, rocblas_int ldb, double* C,
                                  rocblas_int ldc);

rocblas_status rocblas_port_ctrmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
                                  rocblas_diagonal diag, rocblas_int m, rocblas_int n, const hipFloatComplex* alpha,
                                  const hipFloatComplex* A, rocblas_int lda, const hipFloatComplex* B, rocblas_int ldb,
                                  hipFloatComplex* C, rocblas_int ldc);

rocblas_status rocblas_port_ztrmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
                                  rocblas_diagonal diag, rocblas_int m, rocblas_int n, const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* A, rocblas_int lda, const hipDoubleComplex* B,
                                  rocblas_int ldb, hipDoubleComplex* C, rocblas_int ldc);
/*
 * GER
 */

rocblas_status rocblas_port_sger(rocblas_handle handle, rocblas_int m, rocblas_int n, const float* alpha,
                                 const float* x, rocblas_int incx, const float* y, rocblas_int incy, float* A,
                                 rocblas_int lda);

rocblas_status rocblas_port_dger(rocblas_handle handle, rocblas_int m, rocblas_int n, const double* alpha,
                                 const double* x, rocblas_int incx, const double* y, rocblas_int incy, double* A,
                                 rocblas_int lda);

rocblas_status rocblas_port_cgeru(rocblas_handle handle, rocblas_int m, rocblas_int n, const hipFloatComplex* alpha,
                                  const hipFloatComplex* x, rocblas_int incx, const hipFloatComplex* y,
                                  rocblas_int incy, hipFloatComplex* A, rocblas_int lda);

rocblas_status rocblas_port_zgeru(rocblas_handle handle, rocblas_int m, rocblas_int n, const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* x, rocblas_int incx, const hipDoubleComplex* y,
                                  rocblas_int incy, hipDoubleComplex* A, rocblas_int lda);

/*
 * AXPY
 */
rocblas_status rocblas_port_saxpy(rocblas_handle handle, rocblas_int n, const float* alpha, const float* x,
                                  rocblas_int incx, float* y, rocblas_int incy);

rocblas_status rocblas_port_daxpy(rocblas_handle handle, rocblas_int n, const double* alpha, const double* x,
                                  rocblas_int incx, double* y, rocblas_int incy);

rocblas_status rocblas_port_caxpy(rocblas_handle handle, rocblas_int n, const hipFloatComplex* alpha,
                                  const hipFloatComplex* x, rocblas_int incx, hipFloatComplex* y,
                                  rocblas_int incy);

rocblas_status rocblas_port_zaxpy(rocblas_handle handle, rocblas_int n, const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* x, rocblas_int incx, hipDoubleComplex* y,
                                  rocblas_int incy);

} // extern "C"

#endif
