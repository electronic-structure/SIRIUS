#include <cuda.h>
#include <cublas_v2.h>
#include <map>
#include <string>
#include <vector>
#include <stdio.h>
#include <execinfo.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>
#include <cufft.h>
#include <stdint.h>
#include "Kernels/kernels_common.h"

inline void stack_backtrace()
{
    void *array[10];
    char **strings;
    int size = backtrace(array, 10);
    strings = backtrace_symbols(array, size);
    printf ("Stack backtrace:\n");
    for (size_t i = 0; i < size; i++) printf ("%s\n", strings[i]);
    raise(SIGQUIT);
}

#ifdef NDEBUG
#define CALL_CUDA(func__, args__)                                                                                  \
{                                                                                                                  \
    cudaError_t error = func__ args__;                                                                             \
    if (error != cudaSuccess)                                                                                      \
    {                                                                                                              \
        char nm[1024];                                                                                             \
        gethostname(nm, 1024);                                                                                     \
        printf("hostname: %s\n", nm);                                                                              \
        printf("Error in %s at line %i of file %s: %s\n", #func__, __LINE__, __FILE__, cudaGetErrorString(error)); \
        stack_backtrace();                                                                                         \
    }                                                                                                              \
}
#else
#define CALL_CUDA(func__, args__)                                                                                  \
{                                                                                                                  \
    cudaError_t error;                                                                                             \
    func__ args__;                                                                                                 \
    cudaDeviceSynchronize();                                                                                       \
    error = cudaGetLastError();                                                                                    \
    if (error != cudaSuccess)                                                                                      \
    {                                                                                                              \
        char nm[1024];                                                                                             \
        gethostname(nm, 1024);                                                                                     \
        printf("hostname: %s\n", nm);                                                                              \
        printf("Error in %s at line %i of file %s: %s\n", #func__, __LINE__, __FILE__, cudaGetErrorString(error)); \
        stack_backtrace();                                                                                         \
    }                                                                                                              \
}
#endif

extern "C" {

void cuda_initialize();

void cuda_device_info();

void* cuda_malloc(size_t size);

void* cuda_malloc_host(size_t size);

void cuda_free_host(void* ptr);

void cuda_free(void* ptr);

void cuda_copy_to_device(void* target, void const* source, size_t size);

void cuda_copy_to_host(void* target, void const* source, size_t size);

void cuda_copy_device_to_device(void* target, void const* source, size_t size);

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

void cublas_zgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                  cuDoubleComplex* alpha, cuDoubleComplex* a, int32_t lda, cuDoubleComplex* b, 
                  int32_t ldb, cuDoubleComplex* beta, cuDoubleComplex* c, int32_t ldc, int stream_id);

}
