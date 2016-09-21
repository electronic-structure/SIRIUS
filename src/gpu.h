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

/** \file gpu.h
 *   
 *  \brief Interface to CUDA related functions.
 */

#include <complex>

#ifndef __GPU_H__
#define __GPU_H__

typedef std::complex<double> cuDoubleComplex;

/* cufftHandle is a handle type used to store and access CUFFT plans. */
typedef int cufftHandle;

extern "C" {

void cuda_initialize();

void cuda_device_info();

void* cuda_malloc(size_t size);

void* cuda_malloc_host(size_t size);

void cuda_free_host(void* ptr);

void cuda_free(void* ptr);

void cuda_memset(void *ptr, int value, size_t size);

void cuda_host_register(void* ptr, size_t size);

void cuda_host_unregister(void* ptr);

void cuda_device_synchronize();

void cuda_create_streams(int num_streams);

void cuda_destroy_streams();

int get_num_cuda_streams();

void cuda_stream_synchronize(int stream_id);

void cuda_copy_to_device(void* target, void const* source, size_t size);
void cuda_async_copy_to_device(void* target, void const* source, size_t size, int stream_id);
void cuda_copy2d_to_device(void* dst__, size_t ld1__, const void* src__, size_t ld2__, size_t nrow__,
                           size_t ncol__, int elem_size__);
void cuda_async_copy2d_to_device(void* dst__, size_t ld1__, const void* src__, size_t ld2__,
                                 size_t nrow__, size_t ncol__, int elem_size__, int stream_id__);

void cuda_copy_to_host(void* target, void const* source, size_t size);
void cuda_async_copy_to_host(void* target, void const* source, size_t size, int stream_id);
void cuda_copy2d_to_host(void* dst__, size_t ld1__, const void* src__, size_t ld2__, size_t nrow__, 
                         size_t ncol__, int elem_size__);
void cuda_async_copy2d_to_host(void* dst__, size_t ld1__, const void* src__, size_t ld2__,
                               size_t nrow__, size_t ncol__, int elem_size__, int stream_id__);

void cuda_copy_device_to_device(void* target, void const* source, size_t size);
void cuda_copy2d_device_to_device(void* dst__, size_t ld1__, const void* src__, size_t ld2__,
                                  size_t nrow__, size_t ncol__, int elem_size__);
void cuda_async_copy2d_device_to_device(void* dst__, size_t ld1__, const void* src__, size_t ld2__, size_t nrow__,
                                        size_t ncol__, int elem_size__, int stream_id__);

size_t cuda_get_free_mem();

void cuda_device_reset();

void cuda_check_last_error();

bool cuda_check_device_ptr(void const* ptr__);

void cublas_zgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                  cuDoubleComplex* alpha, cuDoubleComplex const* a, int32_t lda, cuDoubleComplex const* b, 
                  int32_t ldb, cuDoubleComplex* beta, cuDoubleComplex* c, int32_t ldc, int stream_id);

void cublas_dgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                  double* alpha, double const* a, int32_t lda, double const* b, 
                  int32_t ldb, double* beta, double* c, int32_t ldc, int stream_id);

void cublas_dtrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__,
                  double* alpha__, double* A__, int lda__, double* B__, int ldb__);

void cublas_ztrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__,
                  cuDoubleComplex* alpha__, cuDoubleComplex* A__, int lda__, cuDoubleComplex* B__, int ldb__);

void cublas_dger(int m, int n, double* alpha, double* x, int incx, double* y, int incy, double* A, int lda);


void cublas_create_handles(int num_handles);

void cublas_destroy_handles(int num_handles);

void cublas_zgemv(int transa, int32_t m, int32_t n, cuDoubleComplex* alpha, cuDoubleComplex* a, int32_t lda, 
                  cuDoubleComplex* x, int32_t incx, cuDoubleComplex* beta, cuDoubleComplex* y, int32_t incy, 
                  int stream_id);

void cufft_create_plan_handle(cufftHandle* plan);

void cufft_destroy_plan_handle(cufftHandle plan);

size_t cufft_get_work_size(int ndim, int* dims, int nfft);

size_t cufft_create_batch_plan(cufftHandle plan, int rank, int* dims, int* embed, int stride, int dist, int nfft, int auto_alloc);

void cufft_set_work_area(cufftHandle plan, void* work_area);

void cufft_set_stream(cufftHandle plan__, int stream_id__);

void cufft_forward_transform(cufftHandle plan, cuDoubleComplex* fft_buffer);

void cufft_backward_transform(cufftHandle plan, cuDoubleComplex* fft_buffer);

void scale_matrix_columns_gpu(int nrow, int ncol, void* mtrx, double* a);

void scale_matrix_rows_gpu(int nrow, int ncol, void* mtrx, double* v);

#ifdef __MAGMA
void magma_init_wrapper();

void magma_finalize_wrapper();

int magma_dpotrf_wrapper(char uplo, int n, double* A, int lda);

int magma_zpotrf_wrapper(char uplo, int n, cuDoubleComplex* A, int lda);

int magma_dtrtri_wrapper(char uplo, int n, double* A, int lda);

int magma_ztrtri_wrapper(char uplo, int n, cuDoubleComplex* A, int lda);
#endif

void cuda_label_event_gpu(const char* label__);

}

namespace acc {

/// Copy memory inside device.
template <typename T>
inline void copy(T* target__, T const* source__, size_t n__)
{
    cuda_copy_device_to_device(target__, source__, n__ * sizeof(T));
}

/// Copy memory to device.
template <typename T>
inline void copyin(T* target__, T const* source__, size_t n__)
{
    cuda_copy_to_device(target__, source__, n__ * sizeof(T));
}

template <typename T>
inline void copyin(T* target__, T const* source__, size_t n__, int stream_id__)
{
    cuda_async_copy_to_device(target__, source__, n__ * sizeof(T), stream_id__);
}

template <typename T>
inline void copyin(T* target__, int ld1__, T const* source__, int ld2__, int nrow__, int ncol__, int stream_id__)
{
    cuda_async_copy2d_to_device(target__, ld1__, source__, ld2__, nrow__, ncol__, sizeof(T), stream_id__);
}

template <typename T>
inline void copyin(T* target__, int ld1__, T const* source__, int ld2__, int nrow__, int ncol__)
{
    cuda_copy2d_to_device(target__, ld1__, source__, ld2__, nrow__, ncol__, sizeof(T));
}

template <typename T>
inline void copyout(T* target__, T const* source__, size_t n__)
{
    cuda_copy_to_host(target__, source__, n__ * sizeof(T));
}

template <typename T>
inline void copyout(T* target__, T const* source__, size_t n__, int stream_id__)
{
    cuda_async_copy_to_host(target__, source__, n__ * sizeof(T), stream_id__);
}

template <typename T>
inline void copyout(T* target__, int ld1__, T const* source__, int ld2__, int nrow__, int ncol__, int stream_id__)
{
    cuda_async_copy2d_to_host(target__, ld1__, source__, ld2__, nrow__, ncol__, sizeof(T), stream_id__);
}

template <typename T>
inline void copyout(T* target__, int ld1__, T const* source__, int ld2__, int nrow__, int ncol__)
{
    cuda_copy2d_to_host(target__, ld1__, source__, ld2__, nrow__, ncol__, sizeof(T));
}

inline void sync_stream(int stream_id__)
{
    cuda_stream_synchronize(stream_id__);
}

template <typename T>
inline void zero(T* target__, size_t n__)
{
    cuda_memset(target__, 0, n__ * sizeof(T));
}

};

#endif
