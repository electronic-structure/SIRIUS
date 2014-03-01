#ifndef _GPU_INTERFACE_H_
#define _GPU_INTERFACE_H_

#include <stdlib.h>

//================
// CUDA functions
//================

extern "C" void cuda_initialize();

extern "C" void cuda_device_info();

extern "C" void cuda_malloc(void** ptr, size_t size);

extern "C" void cuda_malloc_host(void** ptr, size_t size);

extern "C" void cuda_free_host(void** ptr);

extern "C" void cuda_free(void* ptr);

extern "C" void cuda_copy_to_device(void *target, void *source, size_t size);

extern "C" void cuda_copy_to_host(void *target, void *source, size_t size);

extern "C" void cuda_memset(void *ptr, int value, size_t size);

extern "C" void cuda_host_register(void* ptr, size_t size);

extern "C" void cuda_host_unregister(void* ptr);

extern "C" void cuda_device_synchronize();

extern "C" void cuda_create_streams(int num_streams);

extern "C" void cuda_destroy_streams(int num_streams);

extern "C" void cuda_stream_synchronize(int stream_id);

extern "C" void cuda_async_copy_to_device(void *target, void *source, size_t size, int stream_id);

extern "C" void cuda_async_copy_to_host(void *target, void *source, size_t size, int stream_id);

extern "C" size_t cuda_get_free_mem();

extern "C" void cuda_device_reset();

//==================
// CUBLAS functions
//==================

extern "C" void cublas_init();

extern "C" void cublas_set_stream(int stream_id);

extern "C" void cublas_zgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                             const void* alpha, void* a, int32_t lda, void* b, 
                             int32_t ldb, const void* beta, void* c, int32_t ldc);

extern "C" void cublas_get_matrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);

extern "C" void cublas_get_matrix_async(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, int stream_id);

extern "C" void cublas_set_matrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);

extern "C" void cublas_set_matrix_async(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, int stream_id);

extern "C" void cublas_set_vector(int n, int elemSize, const void *x, int incx, void *y, int incy);

//=================
// CUFFT functions
//=================

extern "C" void cufft_create_batch_plan(int nx, int ny, int nz, int nfft);

extern "C" void cufft_destroy_batch_plan();

extern "C" void cufft_forward_transform(void* fft_buffer);

extern "C" void cufft_backward_transform(void* fft_buffer);

extern "C" void cufft_batch_load_gpu(int num_elements, int* map, void* data, void* fft_buffer);

extern "C" void cufft_batch_unload_gpu(int num_elements, int* map, void* fft_buffer, void* data);

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

