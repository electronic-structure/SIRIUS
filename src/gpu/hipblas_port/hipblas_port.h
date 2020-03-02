#ifndef _HIPBLAS_PORT_H_
#define _HIPBLAS_PORT_H_

#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <hipblas.h>

/*
 * GEMV
 */
hipblasStatus_t hipblas_port_Sgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const float* alpha,
                                   const float* A, int lda, const float* x, int incx, const float* beta, float* y,
                                   int incy);

hipblasStatus_t hipblas_port_Dgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const double* alpha,
                                   const double* A, int lda, const double* x, int incx, const double* beta, double* y,
                                   int incy);

hipblasStatus_t hipblas_port_Cgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n,
                                   const hipFloatComplex* alpha, const hipFloatComplex* A, int lda,
                                   const hipFloatComplex* x, int incx, const hipFloatComplex* beta, hipFloatComplex* y,
                                   int incy);

hipblasStatus_t hipblas_port_Zgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n,
                                   const hipDoubleComplex* alpha, const hipDoubleComplex* A, int lda,
                                   const hipDoubleComplex* x, int incx, const hipDoubleComplex* beta,
                                   hipDoubleComplex* y, int incy);

/*
 * GEMM
 */
hipblasStatus_t hipblas_port_Sgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k, const float *alpha, 
                           const float *A, int lda, 
                           const float *B, int ldb, 
                           const float *beta, 
                           float *C, int ldc);

hipblasStatus_t hipblas_port_Dgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k, const double *alpha, 
                           const double *A, int lda, 
                           const double *B, int ldb, 
                           const double *beta, 
                           double *C, int ldc);

hipblasStatus_t hipblas_port_Cgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k, const hipFloatComplex *alpha, 
                           const hipFloatComplex *A, int lda, 
                           const hipFloatComplex *B, int ldb, 
                           const hipFloatComplex *beta, 
                           hipFloatComplex *C, int ldc);

hipblasStatus_t hipblas_port_Zgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k, const hipDoubleComplex *alpha, 
                           const hipDoubleComplex *A, int lda, 
                           const hipDoubleComplex *B, int ldb, 
                           const hipDoubleComplex *beta, 
                           hipDoubleComplex *C, int ldc);

/*
 * TRMM
 */

hipblasStatus_t hipblas_port_Strmm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo,
                                   hipblasOperation_t trans, hipblasDiagType_t diag, int m, int n, const float* alpha,
                                   const float* A, int lda, const float* B, int ldb, float* C, int ldc);

hipblasStatus_t hipblas_port_Dtrmm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo,
                                   hipblasOperation_t trans, hipblasDiagType_t diag, int m, int n, const double* alpha,
                                   const double* A, int lda, const double* B, int ldb, double* C, int ldc);

hipblasStatus_t hipblas_port_Ctrmm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo,
                                   hipblasOperation_t trans, hipblasDiagType_t diag, int m, int n,
                                   const hipFloatComplex* alpha, const hipFloatComplex* A, int lda,
                                   const hipFloatComplex* B, int ldb, hipFloatComplex* C, int ldc);

hipblasStatus_t hipblas_port_Ztrmm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo,
                                   hipblasOperation_t trans, hipblasDiagType_t diag, int m, int n,
                                   const hipDoubleComplex* alpha, const hipDoubleComplex* A, int lda,
                                   const hipDoubleComplex* B, int ldb, hipDoubleComplex* C, int ldc);



/*
 * GER
 */
hipblasStatus_t hipblas_port_Sger(hipblasHandle_t handle, int m, int n, const float* alpha, const float* x, int incx,
                                  const float* y, int incy, float* A, int lda);

hipblasStatus_t hipblas_port_Dger(hipblasHandle_t handle, int m, int n, const double* alpha, const double* x, int incx,
                                  const double* y, int incy, double* A, int lda);

hipblasStatus_t hipblas_port_Cgeru(hipblasHandle_t handle, int m, int n, const hipFloatComplex* alpha,
                                   const hipFloatComplex* x, int incx, const hipFloatComplex* y, int incy,
                                   hipFloatComplex* A, int lda);

hipblasStatus_t hipblas_port_Zgeru(hipblasHandle_t handle, int m, int n, const hipDoubleComplex* alpha,
                                   const hipDoubleComplex* x, int incx, const hipDoubleComplex* y, int incy,
                                   hipDoubleComplex* A, int lda);

/*
 * AXPY
 */
hipblasStatus_t hipblas_port_Saxpy(hipblasHandle_t handle, int n, const float* alpha, const float* x, int incx,
                                   float* y, int incy);

hipblasStatus_t hipblas_port_Daxpy(hipblasHandle_t handle, int n, const double* alpha, const double* x, int incx,
                                   double* y, int incy);

hipblasStatus_t hipblas_port_Caxpy(hipblasHandle_t handle, int n, const hipFloatComplex* alpha,
                                   const hipFloatComplex* x, int incx, hipFloatComplex* y, int incy);

hipblasStatus_t hipblas_port_Zaxpy(hipblasHandle_t handle, int n, const hipDoubleComplex* alpha,
                                   const hipDoubleComplex* x, int incx, hipDoubleComplex* y, int incy);
#endif
