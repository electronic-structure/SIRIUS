#ifndef _GPU_INTERFACE_H_
#define _GPU_INTERFACE_H_

extern "C" void cuda_init();

extern "C" void cublas_init();

extern "C" void cuda_malloc_host(void** ptr, size_t size);

extern "C" void cuda_free_host(void** ptr);


extern "C" void cuda_malloc(void** ptr, int size);

extern "C" void cuda_free(void* ptr);

extern "C" void cuda_copy_to_device(void *target, void *source, int size);

extern "C" void cuda_copy_to_host(void *target, void *source, int size);

extern "C" void cuda_memset(void *ptr, int value, size_t size);

extern "C" void magma_zhegvdx_2stage_wrapper(int32_t matrix_size, int32_t nv, void* a, int32_t lda, 
                                             void* b, int32_t ldb, double* eval);

#endif // _GPU_INTERFACE_H_

