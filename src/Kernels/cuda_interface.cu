#include <cuda.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <execinfo.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>
#include "kernels_common.h"

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

//================
// CUDA functions
//================

std::vector<cudaStream_t> cuda_streams;

cudaStream_t cuda_stream_by_id(int stream_id__)
{
    return (stream_id__ == -1) ? NULL : cuda_streams[stream_id__];
}

extern "C" {

// Free memory in bytes
size_t cuda_get_free_mem()
{
    size_t free, total;
    CALL_CUDA(cudaMemGetInfo, (&free, &total));

    return free;
}

void cuda_device_info()
{
    int count;
    CALL_CUDA(cudaGetDeviceCount, (&count));

    if (count == 0)
    {
        printf("CUDA devices not found\n");
        exit(-100);
    }

    cudaDeviceProp devprop;
     
    CALL_CUDA(cudaGetDeviceProperties, (&devprop, 0));
    
    printf("name                        : %s \n",       devprop.name);
    printf("major                       : %i \n",       devprop.major);
    printf("minor                       : %i \n",       devprop.minor);
    printf("asyncEngineCount            : %i \n",       devprop.asyncEngineCount);
    printf("canMapHostMemory            : %i \n",       devprop.canMapHostMemory);
    printf("clockRate                   : %i kHz \n",   devprop.clockRate);
    printf("concurrentKernels           : %i \n",       devprop.concurrentKernels);
    printf("ECCEnabled                  : %i \n",       devprop.ECCEnabled);
    printf("l2CacheSize                 : %i kB \n",    devprop.l2CacheSize / 1024);
    printf("maxGridSize                 : %i %i %i \n", devprop.maxGridSize[0], devprop.maxGridSize[1], devprop.maxGridSize[2]);
    printf("maxThreadsDim               : %i %i %i \n", devprop.maxThreadsDim[0], devprop.maxThreadsDim[1], devprop.maxThreadsDim[2]);
    printf("maxThreadsPerBlock          : %i \n",       devprop.maxThreadsPerBlock);
    printf("maxThreadsPerMultiProcessor : %i \n",       devprop.maxThreadsPerMultiProcessor);
    printf("memoryBusWidth              : %i bits \n",  devprop.memoryBusWidth);
    printf("memoryClockRate             : %i kHz \n",   devprop.memoryClockRate);
    printf("memPitch                    : %zi \n",      devprop.memPitch);
    printf("multiProcessorCount         : %i \n",       devprop.multiProcessorCount);
    printf("regsPerBlock                : %i \n",       devprop.regsPerBlock);
    printf("sharedMemPerBlock           : %li kB \n",   devprop.sharedMemPerBlock / 1024);
    printf("totalConstMem               : %li kB \n",   devprop.totalConstMem / 1024);
    printf("totalGlobalMem              : %li kB \n",   devprop.totalGlobalMem / 1024);
    printf("available memory            : %li kB \n",   cuda_get_free_mem() / 1024);
}

void print_cuda_timers()
{
    CUDA_timer::cuda_timers_wrapper().print();
}

void cuda_initialize()
{
    //CALL_CUDA(cudaSetDeviceFlags, (cudaDeviceMapHost));
}

/* 
 * memory allocation functions 
 */
void* cuda_malloc(size_t size)
{
    void* ptr;
    CALL_CUDA(cudaMalloc, (&ptr, size));
    return ptr;
}

void cuda_free(void* ptr)
{
    CALL_CUDA(cudaFree, (ptr));
}

void* cuda_malloc_host(size_t size)
{
    void* ptr;
    CALL_CUDA(cudaMallocHost, (&ptr, size));
    return ptr;
}

void cuda_free_host(void* ptr)
{
    CALL_CUDA(cudaFreeHost, (ptr));
}

void cuda_device_synchronize()
{
    CALL_CUDA(cudaDeviceSynchronize, ());
}

void cuda_device_reset()
{
    CALL_CUDA(cudaDeviceReset, ());
}

void cuda_create_streams(int num_streams__)
{
    cuda_streams = std::vector<cudaStream_t>(num_streams__);

    //for (int i = 0; i < num_streams; i++) cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    for (size_t i = 0; i < cuda_streams.size(); i++) {
        CALL_CUDA(cudaStreamCreate, (&cuda_streams[i]));
    }
}

int get_num_cuda_streams()
{
    return static_cast<int>(cuda_streams.size());
}

void cuda_destroy_streams()
{
    for (size_t i = 0; i < cuda_streams.size(); i++) {
        CALL_CUDA(cudaStreamDestroy, (cuda_streams[i]));
    }
}

void cuda_stream_synchronize(int stream_id__)
{
    CALL_CUDA(cudaStreamSynchronize, (cuda_stream_by_id(stream_id__)));
}

void cuda_memset(void* ptr, int value, size_t size)
{
    CALL_CUDA(cudaMemset, (ptr, value, size));
}

void cuda_host_register(void* ptr, size_t size)
{
    assert(ptr);
    
    CALL_CUDA(cudaHostRegister, (ptr, size, cudaHostRegisterMapped));
}

void cuda_host_unregister(void* ptr)
{
    CALL_CUDA(cudaHostUnregister, (ptr));
}

void cuda_check_last_error()
{
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error != cudaSuccess\n");
    }
}

void cuda_copy_to_device(void* target, void const* source, size_t size)
{
    CALL_CUDA(cudaMemcpy, (target, source, size, cudaMemcpyHostToDevice));
}

void cuda_copy_to_host(void* target, void const* source, size_t size)
{
    CALL_CUDA(cudaMemcpy, (target, source, size, cudaMemcpyDeviceToHost));
}

void cuda_copy_device_to_device(void* target, void const* source, size_t size)
{
    CALL_CUDA(cudaMemcpy, (target, source, size, cudaMemcpyDeviceToDevice));
}

void cuda_async_copy_to_device(void* target, void* source, size_t size, int stream_id)
{
    CALL_CUDA(cudaMemcpyAsync, (target, source, size, cudaMemcpyHostToDevice, cuda_stream_by_id(stream_id)));
}

void cuda_async_copy_to_host(void* target, void* source, size_t size, int stream_id)
{
    CALL_CUDA(cudaMemcpyAsync, (target, source, size, cudaMemcpyDeviceToHost, cuda_stream_by_id(stream_id)));
}

void cuda_async_copy_device_to_device(void* target, void const* source, size_t size, int stream_id)
{
    CALL_CUDA(cudaMemcpyAsync, (target, source, size, cudaMemcpyDeviceToDevice, cuda_stream_by_id(stream_id)));
}

void cuda_copy2d_device_to_device(void* dst__, size_t ld1__, const void* src__, size_t ld2__,
                                  size_t nrow__, size_t ncol__, int elem_size__)
{
    CALL_CUDA(cudaMemcpy2D, (dst__, ld1__ * elem_size__, src__, ld2__ * elem_size__,
                             nrow__ * elem_size__, ncol__, cudaMemcpyDeviceToDevice));
}

void cuda_async_copy2d_device_to_device(void* dst__, size_t ld1__, const void* src__, size_t ld2__, size_t nrow__, size_t ncol__, int elem_size__, int stream_id__)
{
    CALL_CUDA(cudaMemcpy2DAsync, (dst__, ld1__ * elem_size__, src__, ld2__ * elem_size__, nrow__ * elem_size__, ncol__, cudaMemcpyDeviceToDevice, cuda_stream_by_id(stream_id__)));
}

void cuda_copy2d_to_device(void* dst__, size_t ld1__, const void* src__, size_t ld2__, size_t nrow__,
                           size_t ncol__, int elem_size__)
{
    CALL_CUDA(cudaMemcpy2D, (dst__, ld1__ * elem_size__, src__, ld2__ * elem_size__, nrow__ * elem_size__, ncol__, cudaMemcpyHostToDevice));
}

void cuda_async_copy2d_to_device(void* dst__, size_t ld1__, const void* src__, size_t ld2__, size_t nrow__,
                                 size_t ncol__, int elem_size__, int stream_id__)
{
    CALL_CUDA(cudaMemcpy2DAsync, (dst__, ld1__ * elem_size__, src__, ld2__ * elem_size__, nrow__ * elem_size__, ncol__, cudaMemcpyHostToDevice, cuda_stream_by_id(stream_id__)));
}

void cuda_async_copy2d_to_host(void* dst__, size_t ld1__, const void* src__, size_t ld2__, size_t nrow__, 
                               size_t ncol__, int elem_size__, int stream_id__)
{
    CALL_CUDA(cudaMemcpy2DAsync, (dst__, ld1__ * elem_size__, src__, ld2__ * elem_size__, nrow__ * elem_size__, ncol__, cudaMemcpyDeviceToHost, cuda_stream_by_id(stream_id__)));
}

void cuda_copy2d_to_host(void* dst__, size_t ld1__, const void* src__, size_t ld2__, size_t nrow__, 
                         size_t ncol__, int elem_size__)
{
    CALL_CUDA(cudaMemcpy2D, (dst__, ld1__ * elem_size__, src__, ld2__ * elem_size__, nrow__ * elem_size__, ncol__, cudaMemcpyDeviceToHost));
}

bool cuda_check_device_ptr(void const* ptr__)
{
    cudaPointerAttributes attr;
    cudaError_t error = cudaPointerGetAttributes(&attr, ptr__);
    cudaGetLastError();
    if (error != cudaSuccess) {
        return false;
    }
    if (attr.memoryType == cudaMemoryTypeDevice) {
        return true;
    }
    return false;
}

void cuda_memset2d(void* ptr__, int ld__, int nrow__, int ncol__, int elem_size__, int value__)
{
    CALL_CUDA(cudaMemset2D, (ptr__, ld__ * elem_size__, value__, nrow__ * elem_size__, ncol__));
}



//int cuda_check_device_ptr(void *ptr__) 
//{
//    int data;
//    CUresult result;
//
//    result = cuPointerGetAttribute(&data, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)ptr__);
//    if (result != CUDA_SUCCESS) {
//        return 0;
//    }
//    return (data == CU_MEMORYTYPE_DEVICE);
//}


} // extern "C"



//==================
// CUBLAS functions
//==================

cublasHandle_t cublas_null_stream_handle;
std::vector<cublasHandle_t> cublas_handles;

void cublas_error_message(cublasStatus_t status)
{
    switch (status)
    {
        case CUBLAS_STATUS_NOT_INITIALIZED:
        {
            printf("the library was not initialized\n");
            break;
        }
        case CUBLAS_STATUS_INVALID_VALUE:
        {
            printf("the parameters m,n,k<0\n");
            break;
        }
        case CUBLAS_STATUS_ARCH_MISMATCH:
        {
            printf("the device does not support double-precision\n");
            break;
        }
        case CUBLAS_STATUS_EXECUTION_FAILED:
        {
            printf("the function failed to launch on the GPU\n");
            break;
        }
        default:
        {
            printf("cublas status unknown");
        }
    }
}

#ifdef NDEBUG
    #define CALL_CUBLAS(func__, args__)                                                 \
    {                                                                                   \
        cublasStatus_t status;                                                          \
        if ((status = func__ args__) != CUBLAS_STATUS_SUCCESS)                          \
        {                                                                               \
            cublas_error_message(status);                                               \
            char nm[1024];                                                              \
            gethostname(nm, 1024);                                                      \
            printf("hostname: %s\n", nm);                                               \
            printf("Error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__); \
            stack_backtrace();                                                          \
        }                                                                               \
    }
#else
    #define CALL_CUBLAS(func__, args__)                                                 \
    {                                                                                   \
        cublasStatus_t status;                                                          \
        func__ args__;                                                                  \
        cudaDeviceSynchronize();                                                        \
        status = cublasGetError();                                                      \
        if (status != CUBLAS_STATUS_SUCCESS)                                            \
        {                                                                               \
            cublas_error_message(status);                                               \
            char nm[1024];                                                              \
            gethostname(nm, 1024);                                                      \
            printf("hostname: %s\n", nm);                                               \
            printf("Error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__); \
            stack_backtrace();                                                          \
        }                                                                               \
    }
#endif

extern "C" void cublas_create_handles(int num_handles)
{
    CALL_CUBLAS(cublasCreate, (&cublas_null_stream_handle));
    
    cublas_handles = std::vector<cublasHandle_t>(num_handles);
    for (int i = 0; i < num_handles; i++)
    {
        CALL_CUBLAS(cublasCreate, (&cublas_handles[i]));

        CALL_CUBLAS(cublasSetStream, (cublas_handles[i], cuda_stream_by_id(i)));
    }
}

extern "C" void cublas_destroy_handles(int num_handles)
{
    CALL_CUBLAS(cublasDestroy, (cublas_null_stream_handle));
    for (int i = 0; i < num_handles; i++)
    {
        CALL_CUBLAS(cublasDestroy, (cublas_handles[i]));
    }
}

cublasHandle_t cublas_handle_by_id(int id__)
{
    return (id__ == -1) ? cublas_null_stream_handle : cublas_handles[id__];
}



extern "C" void cublas_zgemv(int transa, int32_t m, int32_t n, cuDoubleComplex* alpha, cuDoubleComplex* a, int32_t lda, 
                             cuDoubleComplex* x, int32_t incx, cuDoubleComplex* beta, cuDoubleComplex* y, int32_t incy, 
                             int stream_id)
{
    const cublasOperation_t trans[] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};

    CALL_CUBLAS(cublasZgemv, (cublas_handle_by_id(stream_id), trans[transa], m, n, alpha, a, lda, x, incx, beta, y, incy));
}

extern "C" void cublas_zgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                             cuDoubleComplex* alpha, cuDoubleComplex* a, int32_t lda, cuDoubleComplex* b, 
                             int32_t ldb, cuDoubleComplex* beta, cuDoubleComplex* c, int32_t ldc, int stream_id)
{
    const cublasOperation_t trans[] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};
    
    CALL_CUBLAS(cublasZgemm, (cublas_handle_by_id(stream_id), trans[transa], trans[transb], m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

extern "C" void cublas_dgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                             double* alpha, double* a, int32_t lda, double* b, 
                             int32_t ldb, double* beta, double* c, int32_t ldc, int stream_id)
{
    const cublasOperation_t trans[] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};
    
    CALL_CUBLAS(cublasDgemm, (cublas_handle_by_id(stream_id), trans[transa], trans[transb], m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

extern "C" void cublas_dtrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__,
                             double* alpha__, double* A__, int lda__, double* B__, int ldb__)
{
    if (!(side__ == 'L' || side__ == 'R'))
    {
        printf("cublas_dtrmm: wrong side\n");
        exit(-1);
    }
    cublasSideMode_t side = (side__ == 'L') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;

    if (!(uplo__ == 'U' || uplo__ == 'L'))
    {
        printf("cublas_dtrmm: wrong uplo\n");
        exit(-1);
    }
    cublasFillMode_t uplo = (uplo__ == 'U') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

    if (!(transa__ == 'N' || transa__ == 'T' || transa__ == 'C'))
    {
        printf("cublas_dtrmm: wrong transa\n");
        exit(-1);
    }
    cublasOperation_t transa = CUBLAS_OP_N;
    if (transa__ == 'T') transa = CUBLAS_OP_T;
    if (transa__ == 'C') transa = CUBLAS_OP_C;

    if (!(diag__ == 'N' || diag__ == 'U'))
    {
        printf("cublas_dtrmm: wrong diag\n");
        exit(-1);
    }
    cublasDiagType_t diag = (diag__ == 'N') ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT;

    CALL_CUBLAS(cublasDtrmm, (cublas_null_stream_handle, side, uplo, transa, diag, m__, n__, alpha__, A__, lda__, B__, ldb__, B__, ldb__));
}

extern "C" void cublas_ztrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__,
                             cuDoubleComplex* alpha__, cuDoubleComplex* A__, int lda__, cuDoubleComplex* B__, int ldb__)
{
    if (!(side__ == 'L' || side__ == 'R'))
    {
        printf("cublas_ztrmm: wrong side\n");
        exit(-1);
    }
    cublasSideMode_t side = (side__ == 'L') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;

    if (!(uplo__ == 'U' || uplo__ == 'L'))
    {
        printf("cublas_ztrmm: wrong uplo\n");
        exit(-1);
    }
    cublasFillMode_t uplo = (uplo__ == 'U') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

    if (!(transa__ == 'N' || transa__ == 'T' || transa__ == 'C'))
    {
        printf("cublas_ztrmm: wrong transa\n");
        exit(-1);
    }
    cublasOperation_t transa = CUBLAS_OP_N;
    if (transa__ == 'T') transa = CUBLAS_OP_T;
    if (transa__ == 'C') transa = CUBLAS_OP_C;

    if (!(diag__ == 'N' || diag__ == 'U'))
    {
        printf("cublas_ztrmm: wrong diag\n");
        exit(-1);
    }
    cublasDiagType_t diag = (diag__ == 'N') ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT;

    CALL_CUBLAS(cublasZtrmm, (cublas_null_stream_handle, side, uplo, transa, diag, m__, n__, alpha__, A__, lda__, B__, ldb__, B__, ldb__));
}

extern "C" void cublas_dger(int m, int n, double* alpha, double* x, int incx, double* y, int incy, double* A, int lda)
{
    CALL_CUBLAS(cublasDger, (cublas_null_stream_handle, m, n, alpha, x, incx, y, incy, A, lda));
}

//== inline __device__ uint32_t random(size_t seed)
//== {
//==     uint32_t h = 5381;
//== 
//==     return (h << (seed % 15)) + h;
//== }
//== 
//== __global__ void randomize_on_gpu_kernel
//== (
//==     double* ptr__,
//==     size_t size__
//== )
//== {
//==     int i = blockIdx.x * blockDim.x + threadIdx.x;
//==     if (i < size__) ptr__[i] = double(random(i)) / (1 << 31);
//== }
//== 
//== extern "C" void randomize_on_gpu(double* ptr, size_t size)
//== {
//==     dim3 grid_t(64);
//==     dim3 grid_b(num_blocks(size, grid_t.x));
//== 
//==     randomize_on_gpu_kernel <<<grid_b, grid_t>>>
//==     (
//==         ptr,
//==         size
//==     );
//== }

__global__ void cuda_label_event_gpu_kernel(const char* label__)
{
}

extern "C" void cuda_label_event_gpu(const char* label__)
{
    dim3 grid_t(32);
    dim3 grid_b(1);
    cuda_label_event_gpu_kernel <<<grid_b, grid_t>>>
    (
        label__
    );
}


