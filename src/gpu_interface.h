#ifndef _GPU_INTERFACE_H_
#define _GPU_INTERFACE_H_

extern "C" void cuda_init();

extern "C" void cublas_init();

extern "C" void cuda_device_info();

extern "C" void cuda_malloc_host(void** ptr, size_t size);

extern "C" void cuda_free_host(void** ptr);


extern "C" void cuda_malloc(void** ptr, size_t size);

extern "C" void cuda_free(void* ptr);

extern "C" void cuda_copy_to_device(void *target, void *source, size_t size);

extern "C" void cuda_copy_to_host(void *target, void *source, size_t size);

extern "C" void cuda_memset(void *ptr, int value, size_t size);


extern "C" void cublas_zgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                             void* alpha, void* a, int32_t lda, void* b, 
                             int32_t ldb, void* beta, void* c, int32_t ldc);

extern "C" void cublas_get_matrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);

extern "C" void cublas_set_matrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);

extern "C" void cuda_host_register(void* ptr, size_t size);

extern "C" void cuda_host_unregister(void* ptr);

#ifdef _MAGMA_
extern "C" void magma_zhegvdx_2stage_wrapper(int32_t matrix_size, int32_t nv, void* a, int32_t lda, 
                                             void* b, int32_t ldb, double* eval);
#endif

extern "C" void cuda_create_streams(int num_streams);

extern "C" void cuda_destroy_streams(int num_streams);

extern "C" void cuda_stream_synchronize(int stream_id);

extern "C" void cuda_async_copy_to_device(void *target, void *source, size_t size, int stream_id);

extern "C" void cuda_async_copy_to_host(void *target, void *source, size_t size, int stream_id);

template <typename T>
void spline_inner_product_gpu(int size, double* r_dr, T* s1_coefs, T* s2_coefs);

void sbessel_vlm_inner_product_gpu(int* kargs, int lmmax_pot, int num_atoms, int* iat_by_ia, int* l_by_lm, 
                                   int* nmtp_by_iat, double* r_dr, double* sbessel_coefs, double* vlm_coefs, 
                                   double* jvlm, int stream_id);

void sbessel_lo_inner_product_gpu(int* kargs, int num_gkvec, int num_lo, int* l_by_ilo, int* iat_by_ilo, 
                                  int* nmtp_by_iat, double* r_dr, double* sbessel_coefs, double* lo_coefs, double* jlo);

void sbessel_vlo_inner_product_gpu(int* kargs, int num_gkvec, int num_lo, int lmmax_pw, int* l_by_lm, int* iat_by_ilo, 
                                   int* nmtp_by_iat, double* r_dr, double* sbessel_coefs, void* vlo_coefs, void* jvlo);
#endif // _GPU_INTERFACE_H_

