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

extern "C" void magma_zhegvdx_2stage_wrapper(int32_t matrix_size, int32_t nv, void* a, int32_t lda, 
                                             void* b, int32_t ldb, double* eval);

extern "C" void cublas_zgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                             void* alpha, void* a, int32_t lda, void* b, 
                             int32_t ldb, void* beta, void* c, int32_t ldc);

extern "C" void cublas_get_matrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);

#endif // _GPU_INTERFACE_H_

