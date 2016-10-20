#include <cuda.h>
#include <cublas_v2.h>
#include <execinfo.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>
#include <nvToolsExt.h>
#include "kernels_common.h"

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

} // extern "C"



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

#ifdef __GPU_NVTX
extern "C" void cuda_begin_range_marker(const char* label__)
{
    nvtxRangePushA(label__);
}

extern "C" void cuda_end_range_marker()
{
    nvtxRangePop();
}
#endif
