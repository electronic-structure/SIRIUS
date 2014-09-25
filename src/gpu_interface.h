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

/** \file gpu_interface.h
 *   
 *  \brief CUDA related functions.
 */

#ifndef _GPU_INTERFACE_H_
#define _GPU_INTERFACE_H_

#include <complex>
#include <cstdlib>

const int _null_stream_ = -1;

//================
// CUDA functions
//================

extern "C" {

void cuda_initialize();

void cuda_device_info();

void* cuda_malloc(size_t size);

void* cuda_malloc_host(size_t size);

void cuda_free_host(void* ptr);

void cuda_free(void* ptr);

void cuda_copy_to_device(void* target, void const* source, size_t size);

void cuda_copy_to_host(void *target, void *source, size_t size);

void cuda_copy_device_to_device(void* target, void* source, size_t size);

void cuda_memset(void *ptr, int value, size_t size);

void cuda_host_register(void* ptr, size_t size);

void cuda_host_unregister(void* ptr);

void cuda_device_synchronize();

void cuda_create_streams(int num_streams);

void cuda_destroy_streams(int num_streams);

void cuda_stream_synchronize(int stream_id);

void cuda_async_copy_to_device(void *target, void *source, size_t size, int stream_id);

void cuda_async_copy_to_host(void *target, void *source, size_t size, int stream_id);

size_t cuda_get_free_mem();

void cuda_device_reset();

void cuda_check_last_error();

}

//==================
// CUBLAS functions
//==================

extern "C" void cublas_create_handles(int num_handles);

extern "C" void cublas_destroy_handles(int num_handles);

//extern "C" void cublas_init();

//extern "C" void cublas_set_stream(int stream_id);

extern "C" void cublas_zgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                             const void* alpha, void* a, int32_t lda, void* b, 
                             int32_t ldb, const void* beta, void* c, int32_t ldc, int stream_id);

extern "C" void cublas_get_matrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);

extern "C" void cublas_get_matrix_async(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, int stream_id);

extern "C" void cublas_set_matrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);

extern "C" void cublas_set_matrix_async(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, int stream_id);

extern "C" void cublas_set_vector(int n, int elemSize, const void *x, int incx, void *y, int incy);

//=================
// CUFFT functions
//=================

// cufftHandle is a handle type used to store and access CUFFT plans.
typedef int cufftHandle;

typedef std::complex<double> cuDoubleComplex;

extern "C" void cufft_create_plan_handle(cufftHandle* plan);

extern "C" void cufft_destroy_plan_handle(cufftHandle plan);

extern "C" size_t cufft_get_size(int nx, int ny, int nz, int nfft);

extern "C" size_t cufft_create_batch_plan(cufftHandle plan, int nx, int ny, int nz, int nfft);

extern "C" void cufft_set_work_area(cufftHandle plan, void* work_area);

extern "C" void cufft_batch_load_gpu(int fft_size,
                                     int num_pw_components, 
                                     int num_fft,
                                     int* map, 
                                     cuDoubleComplex* data, 
                                     cuDoubleComplex* fft_buffer);

extern "C" void cufft_batch_unload_gpu(int fft_size,
                                       int num_pw_components,
                                       int num_fft,
                                       int* map, 
                                       cuDoubleComplex* fft_buffer, 
                                       cuDoubleComplex* data,
                                       double beta);

extern "C" void cufft_forward_transform(cufftHandle plan, cuDoubleComplex* fft_buffer);

extern "C" void cufft_backward_transform(cufftHandle plan, cuDoubleComplex* fft_buffer);

//=================
// MAGMA functions
//=================

#ifdef _MAGMA_
extern "C" void magma_zhegvdx_2stage_wrapper(int32_t matrix_size, int32_t nv, void* a, int32_t lda, 
                                             void* b, int32_t ldb, double* eval);

extern "C" void magma_init_wrapper();

extern "C" void magma_finalize_wrapper();
#endif

//==================================
// High-level functions and kernels
//==================================

template <typename T>
void spline_inner_product_gpu(int size, double* r_dr, T* s1_coefs, T* s2_coefs);

void sbessel_vlm_inner_product_gpu(int* kargs, int lmmax_pot, int num_atoms, int* iat_by_ia, int* l_by_lm, 
                                   int* nmtp_by_iat, double* r_dr, double* sbessel_coefs, double* vlm_coefs, 
                                   double* jvlm, int stream_id);

void sbessel_lo_inner_product_gpu(int* kargs, int num_gkvec, int num_lo, int* l_by_ilo, int* iat_by_ilo, 
                                  int* nmtp_by_iat, double* r_dr, double* sbessel_coefs, double* lo_coefs, double* jlo);

void sbessel_vlo_inner_product_gpu(int* kargs, int num_gkvec, int num_lo, int lmmax_pw, int* l_by_lm, int* iat_by_ilo, 
                                   int* nmtp_by_iat, double* r_dr, double* sbessel_coefs, void* vlo_coefs, void* jvlo);

void add_band_density_gpu(int lmmax_rho, int lmmax_wf, int max_nmtp, int num_atoms_loc, int* ia_by_ialoc, 
                          int* iat_by_ia, int* nmtp_by_iat, int max_num_gaunt, int* gaunt12_size, 
                          int* gaunt12_lm1_by_lm3, int* gaunt12_lm2_by_lm3, void* gaunt12_cg, void* fylm, 
                          double weight, double* dens);

extern "C" void scale_matrix_columns_gpu(int nrow, int ncol, void* mtrx, double* a);

extern "C" void scale_matrix_rows_gpu(int nrow, int ncol, void* mtrx, double* v);


extern "C" void create_beta_pw_gpu(int num_gkvec, 
                                   int num_beta_atot, 
                                   int* beta_t_idx,
                                   void* beta_pw_type,
                                   double* gkvec,
                                   double* atom_pos,
                                   void* beta_pw);

#endif // _GPU_INTERFACE_H_

