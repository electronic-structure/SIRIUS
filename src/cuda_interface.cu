// This file must be compiled with nvcc
#include <execinfo.h>
#include <unistd.h>
#include <signal.h>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <map>
#include <string>
#include <vector>
#include <stdint.h>
#include "cuda_interface.h"

extern "C" void print_cuda_timers()
{
    CUDA_timer::cuda_timers_wrapper().print();
}

//================
// CUDA functions
//================
void stack_backtrace()
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

cudaStream_t* streams;

extern "C" {

void cuda_initialize()
{
    //CALL_CUDA(cudaSetDeviceFlags, (cudaDeviceMapHost));
}

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

void cuda_device_synchronize()
{
    CALL_CUDA(cudaDeviceSynchronize, ());
}

void cuda_device_reset()
{
    CALL_CUDA(cudaDeviceReset, ());
}

void cuda_create_streams(int num_streams)
{
    streams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
    //for (int i = 0; i < num_streams; i++) cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    for (int i = 0; i < num_streams; i++)
    {
        CALL_CUDA(cudaStreamCreate, (&streams[i]));
    }
}

void cuda_destroy_streams(int num_streams)
{
    for (int i = 0; i < num_streams; i++) 
    {
        CALL_CUDA(cudaStreamDestroy, (streams[i]));
    }
    free(streams);
}

void cuda_stream_synchronize(int stream_id)
{
    CALL_CUDA(cudaStreamSynchronize, (streams[stream_id]));
}

void cuda_async_copy_to_device(void* target, void* source, size_t size, int stream_id)
{
    cudaStream_t stream = (stream_id == -1) ? NULL : streams[stream_id];

    CALL_CUDA(cudaMemcpyAsync, (target, source, size, cudaMemcpyHostToDevice, stream));
}

void cuda_async_copy_to_host(void* target, void* source, size_t size, int stream_id)
{
    cudaStream_t stream = (stream_id == -1) ? NULL : streams[stream_id];

    CALL_CUDA(cudaMemcpyAsync, (target, source, size, cudaMemcpyDeviceToHost, stream));
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
    
    printf("name                        : %s \n", devprop.name);
    printf("major                       : %i \n", devprop.major);
    printf("minor                       : %i \n", devprop.minor);
    printf("asyncEngineCount            : %i \n", devprop.asyncEngineCount);
    printf("canMapHostMemory            : %i \n", devprop.canMapHostMemory);
    printf("clockRate                   : %i kHz \n", devprop.clockRate);
    printf("concurrentKernels           : %i \n", devprop.concurrentKernels);
    printf("ECCEnabled                  : %i \n", devprop.ECCEnabled);
    printf("l2CacheSize                 : %i kB \n", devprop.l2CacheSize/1024);
    printf("maxGridSize                 : %i %i %i \n", devprop.maxGridSize[0], devprop.maxGridSize[1], devprop.maxGridSize[2]);
    printf("maxThreadsDim               : %i %i %i \n", devprop.maxThreadsDim[0], devprop.maxThreadsDim[1], devprop.maxThreadsDim[2]);
    printf("maxThreadsPerBlock          : %i \n", devprop.maxThreadsPerBlock);
    printf("maxThreadsPerMultiProcessor : %i \n", devprop.maxThreadsPerMultiProcessor);
    printf("memoryBusWidth              : %i bits \n", devprop.memoryBusWidth);
    printf("memoryClockRate             : %i kHz \n", devprop.memoryClockRate);
    printf("memPitch                    : %zi \n", devprop.memPitch);
    printf("multiProcessorCount         : %i \n", devprop.multiProcessorCount);
    printf("regsPerBlock                : %i \n", devprop.regsPerBlock);
    printf("sharedMemPerBlock           : %li kB \n", devprop.sharedMemPerBlock/1024);
    printf("totalConstMem               : %li kB \n", devprop.totalConstMem/1024);
    printf("totalGlobalMem              : %li kB \n", devprop.totalGlobalMem/1024);
    printf("available memory            : %li kB \n", cuda_get_free_mem() / 1024);
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

}

//==================
// CUBLAS functions
//==================

cublasHandle_t cublas_null_stream_handle;
cublasHandle_t* cublas_stream_handles;

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
        exit(-100);                                                                 \
    }                                                                               \
}

extern "C" void cublas_create_handles(int num_handles)
{
    CALL_CUBLAS(cublasCreate, (&cublas_null_stream_handle));
    
    cublas_stream_handles = (cublasHandle_t*)malloc(num_handles * sizeof(cublasHandle_t));
    for (int i = 0; i < num_handles; i++)
    {
        CALL_CUBLAS(cublasCreate, (&cublas_stream_handles[i]));

        CALL_CUBLAS(cublasSetStream, (cublas_stream_handles[i], streams[i]));
    }
}

extern "C" void cublas_destroy_handles(int num_handles)
{
    CALL_CUBLAS(cublasDestroy, (cublas_null_stream_handle));
    for (int i = 0; i < num_handles; i++)
    {
        CALL_CUBLAS(cublasDestroy, (cublas_stream_handles[i]));
    }
}

//== extern "C" void cublas_set_stream(int stream_id__)
//== {
//==     cudaStream_t stream = (stream_id__ == -1) ? NULL : streams[stream_id__];
//==     cublasSetStream(cublas_handle(), stream);
//== }

extern "C" void cublas_zgemv(int transa, int32_t m, int32_t n, cuDoubleComplex* alpha, cuDoubleComplex* a, int32_t lda, 
                             cuDoubleComplex* x, int32_t incx, cuDoubleComplex* beta, cuDoubleComplex* y, int32_t incy, 
                             int stream_id)
{
    const cublasOperation_t trans[] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};
    cublasHandle_t handle = (stream_id == -1) ? cublas_null_stream_handle : cublas_stream_handles[stream_id];

    CALL_CUBLAS(cublasZgemv, (handle, trans[transa], m, n, alpha, a, lda, x, incx, beta, y, incy));
}

extern "C" void cublas_zgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                             cuDoubleComplex* alpha, cuDoubleComplex* a, int32_t lda, cuDoubleComplex* b, 
                             int32_t ldb, cuDoubleComplex* beta, cuDoubleComplex* c, int32_t ldc, int stream_id)
{
    const cublasOperation_t trans[] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};
    cublasHandle_t handle = (stream_id == -1) ? cublas_null_stream_handle : cublas_stream_handles[stream_id];
    
    CALL_CUBLAS(cublasZgemm, (handle, trans[transa], trans[transb], m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

// A(GPU) => B(CPU)
extern "C" void cublas_get_matrix(int rows, int cols, int elemSize, const void *A_device, int lda, void *B_host, int ldb)
{
    CALL_CUBLAS(cublasGetMatrix, (rows, cols, elemSize, A_device, lda, B_host, ldb));
}

extern "C" void cublas_get_matrix_async(int rows, int cols, int elemSize, const void *A_device, int lda, void *B_host, int ldb, int stream_id)
{
    cudaStream_t stream = (stream_id == -1) ? NULL : streams[stream_id];

    CALL_CUBLAS(cublasGetMatrixAsync, (rows, cols, elemSize, A_device, lda, B_host, ldb, stream));
}

// A(CPU) => B(GPU)
extern "C" void cublas_set_matrix(int rows, int cols, int elemSize, const void *A_host, int lda, void *B_device, int ldb)
{
    CALL_CUBLAS(cublasSetMatrix, (rows, cols, elemSize, A_host, lda, B_device, ldb));
}

extern "C" void cublas_set_matrix_async(int rows, int cols, int elemSize, const void *A_host, int lda, void *B_device, int ldb, int stream_id)
{
    cudaStream_t stream = (stream_id == -1) ? NULL : streams[stream_id];

    CALL_CUBLAS(cublasSetMatrixAsync, (rows, cols, elemSize, A_host, lda, B_device, ldb, stream));
}

// x(CPU) => y(GPU)
extern "C" void cublas_set_vector(int n, int elemSize, const void *x, int incx, void *y, int incy)
{
    CALL_CUBLAS(cublasSetVector, (n, elemSize, x, incx, y, incy));
}

//=================
// CUFFT functions
//=================

void cufft_error_message(cufftResult result)
{
    switch (result)
    {
        case CUFFT_INVALID_PLAN:
        {
            printf("CUFFT_INVALID_PLAN\n");
            break;
        }
        case CUFFT_ALLOC_FAILED:
        {
            printf("CUFFT_ALLOC_FAILED\n");
            break;
        }
        case CUFFT_INVALID_VALUE:
        {
            printf("CUFFT_INVALID_VALUE\n");
            break;
        }
        case CUFFT_INTERNAL_ERROR:
        {
            printf("CUFFT_INTERNAL_ERROR\n");
            break;
        }
        case CUFFT_SETUP_FAILED:
        {
            printf("CUFFT_SETUP_FAILED\n");
            break;
        }
        case CUFFT_INVALID_SIZE:
        {
            printf("CUFFT_INVALID_SIZE\n");
            break;
        }
        default:
        {
            printf("unknown error code %i\n", result);
            break;
        }
    }
}

#define CALL_CUFFT(func__, args__)                                                  \
{                                                                                   \
    cufftResult result;                                                             \
    if ((result = func__ args__) != CUFFT_SUCCESS)                                  \
    {                                                                               \
        char nm[1024];                                                              \
        gethostname(nm, 1024);                                                      \
        printf("hostname: %s\n", nm);                                               \
        printf("Error in %s at line %i of file %s: ", #func__, __LINE__, __FILE__); \
        cufft_error_message(result);                                                \
        exit(-100);                                                                 \
    }                                                                               \
}

extern "C" void cufft_create_plan_handle(cufftHandle* plan)
{
    CALL_CUFFT(cufftCreate, (plan));
}

extern "C" void cufft_destroy_plan_handle(cufftHandle plan)
{
    CALL_CUFFT(cufftDestroy, (plan));
}

/** Get the work size for cuFFT */
extern "C" size_t cufft_get_size(int nx, int ny, int nz, int nfft)
{
    int fft_size = nx * ny * nz;
    int n[] = {nz, ny, nx};
    size_t work_size;

    CALL_CUFFT(cufftEstimateMany, (3, n, NULL, 1, fft_size, NULL, 1, fft_size, CUFFT_Z2Z, nfft, &work_size));
    
    return work_size;
}

extern "C" size_t cufft_create_batch_plan(cufftHandle plan, int nx, int ny, int nz, int nfft)
{
    int fft_size = nx * ny * nz;
    int n[] = {nz, ny, nx};

    CALL_CUFFT(cufftSetAutoAllocation, (plan, false));
    
    size_t work_size;
    CALL_CUFFT(cufftMakePlanMany, (plan, 3, n, n, 1, fft_size, n, 1, fft_size, CUFFT_Z2Z, nfft, &work_size));

    return work_size;
}

extern "C" void cufft_set_work_area(cufftHandle plan, void* work_area)
{
    CALL_CUFFT(cufftSetWorkArea, (plan, work_area));
}

__global__ void cufft_batch_load_gpu_kernel
(
    int fft_size, 
    int num_pw_components, 
    int* map, 
    cuDoubleComplex* data, 
    cuDoubleComplex* fft_buffer
)
{
    int i = blockIdx.y;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_pw_components)
    {
        fft_buffer[array2D_offset(map[idx], i, fft_size)] = data[array2D_offset(idx, i, num_pw_components)];
    }
}

extern "C" void cufft_batch_load_gpu(int fft_size,
                                     int num_pw_components, 
                                     int num_fft,
                                     int* map, 
                                     cuDoubleComplex* data, 
                                     cuDoubleComplex* fft_buffer)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_pw_components, grid_t.x), num_fft);
    
    cuda_memset(fft_buffer, 0, fft_size * num_fft * sizeof(cuDoubleComplex));

    cufft_batch_load_gpu_kernel <<<grid_b, grid_t>>>
    (
        fft_size,
        num_pw_components,
        map,
        data, 
        fft_buffer
    );
}

__global__ void cufft_batch_unload_gpu_kernel
(
    int fft_size, 
    int num_pw_components, 
    int* map, 
    cuDoubleComplex* fft_buffer,
    cuDoubleComplex* data,
    double beta
)
{
    int i = blockIdx.y;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_pw_components) 
    {
        data[array2D_offset(idx, i, num_pw_components)] = cuCadd(
            cuCmul(make_cuDoubleComplex(beta, 0), data[array2D_offset(idx, i, num_pw_components)]),
            cuCdiv(fft_buffer[array2D_offset(map[idx], i, fft_size)], make_cuDoubleComplex(double(fft_size), 0)));
    }
}

extern "C" void cufft_batch_unload_gpu(int fft_size,
                                       int num_pw_components,
                                       int num_fft,
                                       int* map, 
                                       cuDoubleComplex* fft_buffer, 
                                       cuDoubleComplex* data,
                                       double beta)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_pw_components, grid_t.x), num_fft);

    cufft_batch_unload_gpu_kernel <<<grid_b, grid_t>>>
    (
        fft_size, 
        num_pw_components, 
        map, 
        fft_buffer,
        data,
        beta
    );
}

extern "C" void cufft_forward_transform(cufftHandle plan, cuDoubleComplex* fft_buffer)
{
    CUDA_timer t("cufft_forward_transform");
    CALL_CUFFT(cufftExecZ2Z, (plan, fft_buffer, fft_buffer, CUFFT_FORWARD));
}

extern "C" void cufft_backward_transform(cufftHandle plan, cuDoubleComplex* fft_buffer)
{
    CUDA_timer t("cufft_backward_transform");
    CALL_CUFFT(cufftExecZ2Z, (plan, fft_buffer, fft_buffer, CUFFT_INVERSE));
}

//==================================
// High-level functions and kernels
//==================================

template <typename T, typename U>
__device__ U spline_inner_product_gpu_function(int ld, int size, double* r_dr, T* s1_coefs, U* s2_coefs)
{
    int N = size / blockDim.x;
    if (size % blockDim.x != 0) N++;

    extern __shared__ char sdata_ptr[];
    U* sdata = (U*)&sdata_ptr[0];

    int a_offs = 0 * ld;
    int b_offs = 1 * ld;
    int c_offs = 2 * ld;
    int d_offs = 3 * ld;

    sdata[threadIdx.x] = 0;

    for (int n = 0; n < N; n++)
    {
        int i = n * blockDim.x + threadIdx.x;
        if (i < size - 1)
        {
            double x0 = r_dr[i];
            double dx = r_dr[ld + i];

            T a1 = s1_coefs[a_offs + i];
            T b1 = s1_coefs[b_offs + i];
            T c1 = s1_coefs[c_offs + i];
            T d1 = s1_coefs[d_offs + i];
            
            U a2 = s2_coefs[a_offs + i];
            U b2 = s2_coefs[b_offs + i];
            U c2 = s2_coefs[c_offs + i];
            U d2 = s2_coefs[d_offs + i];
                
            U a1a2 = a1 * a2;
            U d1d2 = d1 * d2;
                
            U k1 = d1 * b2 + c1 * c2 + b1 * d2;

            U k2 = d1 * a2 + c1 * b2 + b1 * c2 + a1 * d2;

            U k3 = c1 * a2 + b1 * b2 + a1 * c2;

            U k4 = d1 * c2 + c1 * d2;
            
            U k5 = b1 * a2 + a1 * b2;

            sdata[threadIdx.x] += dx * ((a1a2 * x0 * x0) + 
                                  dx * ((x0 * (2.0 * a1a2 + x0 * k5)) / 2.0 +
                                  dx * ((a1a2 + x0 * (2.0 * k5 + k3 * x0)) / 3.0 + 
                                  dx * ((k5 + x0 * (2.0 * k3 + k2 * x0)) / 4.0 +
                                  dx * ((k3 + x0 * (2.0 * k2 + k1 * x0)) / 5.0 + 
                                  dx * ((k2 + x0 * (2.0 * k1 + k4 * x0)) / 6.0 + 
                                  dx * ((k1 + x0 * (2.0 * k4 + d1d2 * x0)) / 7.0 + 
                                  dx * ((k4 + 2.0 * d1d2 * x0) / 8.0 + 
                                  dx * d1d2 / 9.0)))))))); 
        }
    }
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) 
    {
        if (threadIdx.x % (2 * s) == 0) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    
    //if (threadIdx.x == 0) for (int i = 1; i < blockDim.x; i++) sdata[0] += sdata[i];

    return sdata[0];
}

template <> __device__ 
cuDoubleComplex spline_inner_product_gpu_function<double, cuDoubleComplex>(int ld, int size, double* r_dr, 
                                                                           double* s1_coefs, 
                                                                           cuDoubleComplex* s2_coefs)
{
    int N = size / blockDim.x;
    if (size % blockDim.x != 0) N++;

    extern __shared__ char sdata_ptr[];
    cuDoubleComplex* sdata = (cuDoubleComplex*)&sdata_ptr[0];

    int a_offs = 0 * ld;
    int b_offs = 1 * ld;
    int c_offs = 2 * ld;
    int d_offs = 3 * ld;

    sdata[threadIdx.x] = make_cuDoubleComplex(0.0, 0.0);

    for (int n = 0; n < N; n++)
    {
        int i = n * blockDim.x + threadIdx.x;
        if (i < size - 1)
        {
            double x0 = r_dr[i];
            double dx = r_dr[ld + i];

            double a1 = s1_coefs[a_offs + i];
            double b1 = s1_coefs[b_offs + i];
            double c1 = s1_coefs[c_offs + i];
            double d1 = s1_coefs[d_offs + i];
            
            cuDoubleComplex a2 = s2_coefs[a_offs + i];
            cuDoubleComplex b2 = s2_coefs[b_offs + i];
            cuDoubleComplex c2 = s2_coefs[c_offs + i];
            cuDoubleComplex d2 = s2_coefs[d_offs + i];
                
            cuDoubleComplex a1a2 = make_cuDoubleComplex(a1 * a2.x, a1 * a2.y);
            cuDoubleComplex d1d2 = make_cuDoubleComplex(d1 * d2.x, d1 * d2.y);
                
            cuDoubleComplex k1 = make_cuDoubleComplex(d1 * b2.x + c1 * c2.x + b1 * d2.x, 
                                                      d1 * b2.y + c1 * c2.y + b1 * d2.y);

            cuDoubleComplex k2 = make_cuDoubleComplex(d1 * a2.x + c1 * b2.x + b1 * c2.x + a1 * d2.x, 
                                                      d1 * a2.y + c1 * b2.y + b1 * c2.y + a1 * d2.y);

            cuDoubleComplex k3 = make_cuDoubleComplex(c1 * a2.x + b1 * b2.x + a1 * c2.x, 
                                                      c1 * a2.y + b1 * b2.y + a1 * c2.y);

            cuDoubleComplex k4 = make_cuDoubleComplex(d1 * c2.x + c1 * d2.x, d1 * c2.y + c1 * d2.y);
            
            cuDoubleComplex k5 = make_cuDoubleComplex(b1 * a2.x + a1 * b2.x, b1 * a2.y + a1 * b2.y);

            cuDoubleComplex z = make_cuDoubleComplex(
                                  dx * ((a1a2.x * x0 * x0) + 
                                  dx * ((x0 * (2.0 * a1a2.x + x0 * k5.x)) / 2.0 +
                                  dx * ((a1a2.x + x0 * (2.0 * k5.x + k3.x * x0)) / 3.0 + 
                                  dx * ((k5.x + x0 * (2.0 * k3.x + k2.x * x0)) / 4.0 +
                                  dx * ((k3.x + x0 * (2.0 * k2.x + k1.x * x0)) / 5.0 + 
                                  dx * ((k2.x + x0 * (2.0 * k1.x + k4.x * x0)) / 6.0 + 
                                  dx * ((k1.x + x0 * (2.0 * k4.x + d1d2.x * x0)) / 7.0 + 
                                  dx * ((k4.x + 2.0 * d1d2.x * x0) / 8.0 + 
                                  dx * d1d2.x / 9.0)))))))),
                                  dx * ((a1a2.y * x0 * x0) + 
                                  dx * ((x0 * (2.0 * a1a2.y + x0 * k5.y)) / 2.0 +
                                  dx * ((a1a2.y + x0 * (2.0 * k5.y + k3.y * x0)) / 3.0 + 
                                  dx * ((k5.y + x0 * (2.0 * k3.y + k2.y * x0)) / 4.0 +
                                  dx * ((k3.y + x0 * (2.0 * k2.y + k1.y * x0)) / 5.0 + 
                                  dx * ((k2.y + x0 * (2.0 * k1.y + k4.y * x0)) / 6.0 + 
                                  dx * ((k1.y + x0 * (2.0 * k4.y + d1d2.y * x0)) / 7.0 + 
                                  dx * ((k4.y + 2.0 * d1d2.y * x0) / 8.0 + 
                                  dx * d1d2.y / 9.0)))))))));

            sdata[threadIdx.x] = cuCadd(sdata[threadIdx.x], z);
        }
    }
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) 
    {
        if (threadIdx.x % (2 * s) == 0) sdata[threadIdx.x] = cuCadd(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    
    //if (threadIdx.x == 0) for (int i = 1; i < blockDim.x; i++) sdata[0] = cuCadd(sdata[0], sdata[i]);

    return sdata[0];
}

template <typename T, typename U>
__global__ void spline_inner_product_gpu_kernel(int ld, int size, double* r_dr, T* s1_coefs, U* s2_coefs, U* result)
{
    result[0] = spline_inner_product_gpu_function(ld, size, r_dr, s1_coefs, s2_coefs);
}

template <typename T>
void spline_inner_product_gpu(int size, double* r_dr, T* s1_coefs, T* s2_coefs)
{
    dim3 threadsPerBlock(64);
    dim3 numBlocks(1);

    T* d_result;
    cudaMalloc(&d_result, 1 * sizeof(T));
    spline_inner_product_gpu_kernel<<<numBlocks, threadsPerBlock, 64 * 16>>>(size, size, r_dr, s1_coefs, s2_coefs, d_result);

    T* h_result = (T*)malloc(1 * sizeof(T));
    cudaMemcpy(h_result, d_result, 1 * sizeof(T), cudaMemcpyDeviceToHost);

    printf("GPU result : %18.12f \n", h_result[0]);

    cudaFree(d_result);
    free(h_result);
    
    //cuDoubleComplex* d_zresult;
    //cudaMalloc(&d_zresult, 1 * sizeof(cuDoubleComplex));
    //
    //cuDoubleComplex* zs2;
    //cudaMalloc(&zs2, size * 4 * sizeof(cuDoubleComplex));
    //
    //for (int i = 0; i < size * 4; i++) zs2[i] = make_cuDoubleComplex(s2_coefs[i], s2_coefs[i]);

    //spline_inner_product_gpu_kernel<<<numBlocks, threadsPerBlock, 64 * 16>>>(size, size, r_dr, s1_coefs, zs2, d_zresult);

    //cuDoubleComplex* h_zresult = (cuDoubleComplex*)malloc(1 * sizeof(cuDoubleComplex));
    //cudaMemcpy(h_zresult, d_zresult, 1 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    //printf("GPU result : %18.12f %18.12f\n", h_zresult[0].x, h_zresult[0].y);

    //cudaFree(d_zresult);
    //free(h_zresult);
    //free(zs2);
}

template void spline_inner_product_gpu<double>(int size, double* r_dr, double* s1_coefs, double* s2_coefs);







// Input array dimensions:
//   sbessel_coefs(max_num_mt_points * 4, lmax_pw + 1, num_atom_types, num_gkvec_row);
//   lo_coefs(max_num_mt_points * 4, num_lo);
//   jlo(num_gkvec, num_lo);
__global__ void sbessel_lo_inner_product_gpu_kernel(int* kargs, int num_gkvec, int* l_by_ilo, int* iat_by_ilo, 
                                                    int* nmtp_by_iat, double* r_dr, double* sbessel_coefs, 
                                                    double* lo_coefs, double* jlo)
{
    int num_atom_types = kargs[0];
    int max_nmtp = kargs[1];
    int lmax_pw = kargs[2];

    int igk = blockIdx.x;
    int ilo = blockIdx.y;

    int l = l_by_ilo[ilo];
    int iat = iat_by_ilo[ilo];
    int nmtp = nmtp_by_iat[iat];

    double* jl_ptr = &sbessel_coefs[array4D_offset(0, l, iat, igk, max_nmtp * 4, lmax_pw + 1, num_atom_types)];
    double* lo_ptr = &lo_coefs[array2D_offset(0, ilo, max_nmtp * 4)];
    double* r_dr_ptr = &r_dr[array2D_offset(0, iat, 2 * max_nmtp)];
    
    jlo[array2D_offset(igk, ilo, num_gkvec)] = 
        spline_inner_product_gpu_function(max_nmtp, nmtp, r_dr_ptr, jl_ptr, lo_ptr);
}


void sbessel_lo_inner_product_gpu(int* kargs, int num_gkvec, int num_lo, int* l_by_ilo, int* iat_by_ilo, 
                                  int* nmtp_by_iat, double* r_dr, double* sbessel_coefs, double* lo_coefs, double* jlo)
{
    dim3 threadsPerBlock(64);
    dim3 numBlocks(num_gkvec, num_lo);

    sbessel_lo_inner_product_gpu_kernel<<<numBlocks, threadsPerBlock, 64 * 16>>>
        (kargs, num_gkvec, l_by_ilo, iat_by_ilo, nmtp_by_iat, r_dr, sbessel_coefs, lo_coefs, jlo);
}

// Compute <jl|V|lo>
// Input array dimensions:
//   vlo(max_num_mt_points * 4, lmmax_pw, num_lo_col)
//   jvlo(lmmax_pw, num_gkvec, num_lo)
__global__ void sbessel_vlo_inner_product_gpu_kernel(int* kargs, int num_gkvec, int* l_by_lm, int* iat_by_ilo, 
                                                     int* nmtp_by_iat, double* r_dr, double* sbessel_coefs, 
                                                     cuDoubleComplex* vlo_coefs, cuDoubleComplex* jvlo)
{
    int num_atom_types = kargs[0];
    int max_nmtp = kargs[1];
    int lmax_pw = kargs[2];
    int lmmax_pw = kargs[3];

    int igk = blockIdx.x;
    int ilo = blockIdx.y;
    int lm = blockIdx.z;

    int l = l_by_lm[lm];
    int iat = iat_by_ilo[ilo];
    int nmtp = nmtp_by_iat[iat];
    
    double* jl_ptr = &sbessel_coefs[array4D_offset(0, l, iat, igk, max_nmtp * 4, lmax_pw + 1, num_atom_types)];
    cuDoubleComplex* vlo_ptr = &vlo_coefs[array3D_offset(0, lm, ilo, 4 * max_nmtp, lmmax_pw)];
    double* r_dr_ptr = &r_dr[array2D_offset(0, iat, 2 * max_nmtp)];
    
    jvlo[array3D_offset(lm, igk, ilo, lmmax_pw, num_gkvec)] = 
        spline_inner_product_gpu_function(max_nmtp, nmtp, r_dr_ptr, jl_ptr, vlo_ptr);
}

// Compute <jl|V|lo>
void sbessel_vlo_inner_product_gpu(int* kargs, int num_gkvec, int num_lo, int lmmax_pw, int* l_by_lm, int* iat_by_ilo, 
                                   int* nmtp_by_iat, double* r_dr, double* sbessel_coefs, void* vlo_coefs, void* jvlo)
{
    dim3 threadsPerBlock(64);
    dim3 numBlocks(num_gkvec, num_lo, lmmax_pw);

    sbessel_vlo_inner_product_gpu_kernel<<<numBlocks, threadsPerBlock, 64 * 16>>>
        (kargs, num_gkvec, l_by_lm, iat_by_ilo, nmtp_by_iat, r_dr, sbessel_coefs, (cuDoubleComplex*)vlo_coefs, 
         (cuDoubleComplex*)jvlo);
}

__global__ void sbessel_vlm_inner_product_gpu_kernel(int* kargs, int* iat_by_ia, int* l_by_lm, int* nmtp_by_iat,
                                                     double* r_dr, double* sbessel_coefs, double* vlm_coefs, 
                                                     double* jvlm)
{
    int max_nmtp = kargs[1];
    int lmax_pot = kargs[2];
    int lmmax_pot = kargs[3];
    
    int lm = blockIdx.x;
    int ia = blockIdx.y;

    int iat = iat_by_ia[ia];
    int nmtp = nmtp_by_iat[ia];
    int l = l_by_lm[lm];

    double* jl_ptr = &sbessel_coefs[array3D_offset(0, l, iat, max_nmtp * 4, lmax_pot + 1)];
    double* vlm_ptr = &vlm_coefs[array3D_offset(0, lm, ia, max_nmtp * 4, lmmax_pot)];
    double* r_dr_ptr = &r_dr[array2D_offset(0, iat, 2 * max_nmtp)];

    jvlm[array2D_offset(lm, ia, lmmax_pot)] = 
        spline_inner_product_gpu_function(max_nmtp, nmtp, r_dr_ptr, jl_ptr, vlm_ptr);
}


void sbessel_vlm_inner_product_gpu(int* kargs, int lmmax_pot, int num_atoms, int* iat_by_ia, int* l_by_lm, 
                                   int* nmtp_by_iat, double* r_dr, double* sbessel_coefs, double* vlm_coefs, 
                                   double* jvlm, int stream_id)
{
    cudaStream_t stream = (stream_id == -1) ? NULL : streams[stream_id];
    dim3 threadsPerBlock(64);
    dim3 numBlocks(lmmax_pot, num_atoms);
    
    sbessel_vlm_inner_product_gpu_kernel<<<numBlocks, threadsPerBlock, 64 * 16, stream>>>
        (kargs, iat_by_ia, l_by_lm, nmtp_by_iat, r_dr, sbessel_coefs, vlm_coefs, jvlm);
}


//__global__ void add_band_density_gpu_kernel(int nmtp, int lmmax_rho, int max_nmtp, int max_num_gaunt, int* gaunt12_size, 
//                                            int* gaunt12_lm1_by_lm3, int* gaunt12_lm2_by_lm3, 
//                                            cuDoubleComplex* gaunt12_cg, cuDoubleComplex* fylm, double weight, 
//                                            int ia, double* dens)
//{
//    int ir = blockDim.x * blockIdx.x + threadIdx.x;
//    int lm = blockIdx.y;
//
//    int offs3 = array3D_offset(ir, lm, ia, max_nmtp, lmmax_rho);
//
//    if (ir < nmtp)
//    {
//        for (int k = 0; k < gaunt12_size[lm]; k++)
//        {
//            int offs = array2D_offset(k, lm, max_num_gaunt);
//            int lm1 = gaunt12_lm1_by_lm3[offs];
//            int lm2 = gaunt12_lm2_by_lm3[offs];
//            cuDoubleComplex cg = gaunt12_cg[offs];
//            
//            int offs1 = array2D_offset(ir, lm1, max_nmtp);
//            int offs2 = array2D_offset(ir, lm2, max_nmtp);
//
//            cuDoubleComplex z = cuCmul(cuConj(fylm[offs1]), fylm[offs2]);
//
//            dens[offs3] += weight * cuCreal(cuCmul(z, cg));
//        }
//    }
//}

__global__ void add_band_density_gpu_kernel(int lmmax_rho, int lmmax_wf, int max_nmtp, int* ia_by_ialoc, 
                                            int* iat_by_ia, int* nmtp_by_iat, int max_num_gaunt, 
                                            int* gaunt12_size, int* gaunt12_lm1_by_lm3, int* gaunt12_lm2_by_lm3, 
                                            cuDoubleComplex* gaunt12_cg, cuDoubleComplex* fylm, double weight, 
                                            double* dens)
{
    int lm = blockIdx.x;
    int ialoc = blockIdx.y;
    int ia = ia_by_ialoc[ialoc];
    int iat = iat_by_ia[ia];
    int nmtp = nmtp_by_iat[iat];

    int offs3 = array3D_offset(0, lm, ialoc, max_nmtp, lmmax_rho);

    int N = nmtp / blockDim.x;
    if (nmtp % blockDim.x != 0) N++;

    for (int k = 0; k < gaunt12_size[lm]; k++)
    {
        int offs = array2D_offset(k, lm, max_num_gaunt);

        int lm1 = gaunt12_lm1_by_lm3[offs];
        int lm2 = gaunt12_lm2_by_lm3[offs];
        cuDoubleComplex cg = gaunt12_cg[offs];
        
        int offs1 = array3D_offset(0, lm1, ia, max_nmtp, lmmax_wf);
        int offs2 = array3D_offset(0, lm2, ia, max_nmtp, lmmax_wf);
        
        for (int n = 0; n < N; n++)
        {
            int ir = n * blockDim.x + threadIdx.x;
            if (ir < nmtp)
            {
                cuDoubleComplex z = cuCmul(cuConj(fylm[offs1 + ir]), fylm[offs2 + ir]);

                dens[offs3 + ir] += weight * cuCreal(cuCmul(z, cg));
            }
        }
    }
}

void add_band_density_gpu(int lmmax_rho, int lmmax_wf, int max_nmtp, int num_atoms_loc, int* ia_by_ialoc, 
                          int* iat_by_ia, int* nmtp_by_iat, int max_num_gaunt, int* gaunt12_size, 
                          int* gaunt12_lm1_by_lm3, int* gaunt12_lm2_by_lm3, void* gaunt12_cg, void* fylm, 
                          double weight, double* dens)
{
    dim3 threadsPerBlock(128);
    dim3 numBlocks(lmmax_rho, num_atoms_loc);
    add_band_density_gpu_kernel<<<numBlocks, threadsPerBlock>>>
        (lmmax_rho, lmmax_wf, max_nmtp, ia_by_ialoc, iat_by_ia, nmtp_by_iat, max_num_gaunt, gaunt12_size, 
         gaunt12_lm1_by_lm3, gaunt12_lm2_by_lm3, (cuDoubleComplex*)gaunt12_cg, (cuDoubleComplex*)fylm, weight, dens);
}
    


__global__ void scale_matrix_columns_gpu_kernel
(
    int nrow,
    cuDoubleComplex* mtrx,
    double* a
)
{
    int icol = blockIdx.y;
    int irow = blockIdx.x * blockDim.x + threadIdx.x;
    if (irow < nrow) 
    {
        mtrx[array2D_offset(irow, icol, nrow)] =
            cuCmul(mtrx[array2D_offset(irow, icol, nrow)], make_cuDoubleComplex(a[icol], 0));
    }
}

// scale each column of the matrix by a column-dependent constant
extern "C" void scale_matrix_columns_gpu(int nrow,
                                        int ncol,
                                        cuDoubleComplex* mtrx,
                                        double* a)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(nrow, grid_t.x), ncol);

    scale_matrix_columns_gpu_kernel <<<grid_b, grid_t>>>
    (
        nrow,
        mtrx,
        a
    );
}

__global__ void scale_matrix_rows_gpu_kernel
(
    int nrow,
    cuDoubleComplex* mtrx,
    double* v
)
{
    int icol = blockIdx.y;
    int irow = blockDim.x * blockIdx.x + threadIdx.x;
    if (irow < nrow) 
    {
        mtrx[array2D_offset(irow, icol, nrow)] = 
            cuCmul(mtrx[array2D_offset(irow, icol, nrow)], make_cuDoubleComplex(v[irow], 0));
    }
}

// scale each row of the matrix by a row-dependent constant
extern "C" void scale_matrix_rows_gpu(int nrow,
                                      int ncol,
                                      cuDoubleComplex* mtrx,
                                      double* v)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(nrow, grid_t.x), ncol);

    scale_matrix_rows_gpu_kernel <<<grid_b, grid_t>>>
    (
        nrow,
        mtrx,
        v
    );
}

__global__ void update_it_density_matrix_0_gpu_kernel(int fft_size, 
                                                      int nfft_max, 
                                                      cuDoubleComplex* psi_it, 
                                                      double* wt,
                                                      double* it_density_matrix)
{
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < nfft_max; i++)
    {
        if (ir < fft_size)
        {
            cuDoubleComplex z = psi_it[array3D_offset(ir, i, 0, fft_size, nfft_max)];
            it_density_matrix[array2D_offset(ir, 0, fft_size)] += (z.x * z.x + z.y * z.y) * wt[i];
        }
    }
}

__global__ void update_it_density_matrix_1_gpu_kernel(int fft_size, 
                                                      int nfft_max, 
                                                      cuDoubleComplex* psi_it, 
                                                      double* wt,
                                                      double* it_density_matrix)
{
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < nfft_max; i++)
    {
        if (ir < fft_size)
        {
            cuDoubleComplex z = psi_it[array3D_offset(ir, i, 1, fft_size, nfft_max)];
            it_density_matrix[array2D_offset(ir, 1, fft_size)] += (z.x * z.x + z.y * z.y) * wt[i];
        }
    }
}


extern "C" void update_it_density_matrix_gpu(int fft_size, 
                                             int nfft_max, 
                                             int num_spins, 
                                             int num_mag_dims, 
                                             cuDoubleComplex* psi_it, 
                                             double* wt, 
                                             double* it_density_matrix)
{
    CUDA_timer t("update_it_density_matrix_gpu");

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(fft_size, grid_t.x));

    switch (num_mag_dims)
    {
        //== case 3:
        //== {
        //==     for (int ir = 0; ir < fft_->size(); ir++)
        //==     {
        //==         double_complex z = wfit(ir, 0) * conj(wfit(ir, 1)) * w;
        //==         it_density_matrix(ir, 2) += 2.0 * real(z);
        //==         it_density_matrix(ir, 3) -= 2.0 * imag(z);
        //==     }
        //== }
        case 1:
        {
            update_it_density_matrix_1_gpu_kernel <<<grid_b, grid_t>>>
            (
                fft_size,
                nfft_max,
                psi_it,
                wt,
                it_density_matrix
            );
        }
        case 0:
        {
            update_it_density_matrix_0_gpu_kernel <<<grid_b, grid_t>>>
            (
                fft_size,
                nfft_max,
                psi_it,
                wt,
                it_density_matrix
            );
        }
    }
}
inline __device__ uint32_t random(size_t seed)
{
    uint32_t h = 5381;

    return (h << (seed % 15)) + h;
}

__global__ void randomize_on_gpu_kernel
(
    double* ptr__,
    size_t size__
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size__) ptr__[i] = double(random(i)) / (1 << 31);
}

extern "C" void randomize_on_gpu(double* ptr, size_t size)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(size, grid_t.x));

    randomize_on_gpu_kernel <<<grid_b, grid_t>>>
    (
        ptr,
        size
    );
}


