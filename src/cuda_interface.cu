// This file must be compiled with nvcc

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cufft.h>

//=====================
// Auxiliary functions
//=====================

__device__ size_t array2D_offset(int i0, int i1, int ld0)
{
    return i0 + i1 * ld0;
}

// TODO: can be optimized in terms of multiplication
__device__ size_t array3D_offset(int i0, int i1, int i2, int ld0, int ld1)
{
    return i0 + i1 * ld0 + i2 * ld0 * ld1;
}

// TODO: can be optimized in terms of multiplication
__device__ size_t array4D_offset(int i0, int i1, int i2, int i3, int ld0, int ld1, int ld2)
{
    return i0 + i1 * ld0 + i2 * ld0 * ld1 + i3 * ld0 * ld1 * ld2;
}

int num_blocks(int length, int block_size)
{
    return (length / block_size) + min(length % block_size, 1);
}

//================
// CUDA functions
//================

extern "C" void cuda_malloc(void** ptr, size_t size)
{
    if (cudaMalloc(ptr, size) != cudaSuccess)
    {
        printf("failed to execute cudaMalloc() \n");
        exit(0);
    }
}

extern "C" void cuda_free(void* ptr)
{
    if (cudaFree(ptr) != cudaSuccess)
    {
        printf("failed to execute cudaFree() \n");
        exit(0);
    }
}

extern "C" void cuda_malloc_host(void** ptr, size_t size)
{
    if (cudaMallocHost(ptr, size) != cudaSuccess)
    {  
        printf("cudaMallocHost failed\n");
        exit(-1);
    }
}

extern "C" void cuda_free_host(void** ptr)
{
    if (cudaFreeHost(*ptr) != cudaSuccess)
    {
        printf("cudaFreeHost failed\n");
        exit(-1);
    }
}

extern "C" void cuda_copy_to_device(void* target, void* source, size_t size)
{
    if (cudaMemcpy(target, source, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        printf("failed to execute cudaMemcpy(cudaMemcpyHostToDevice)\n");
        exit(0);
    }
}

extern "C" void cuda_copy_to_host(void* target, void* source, size_t size)
{
    if (cudaMemcpy(target, source, size, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        printf("failed to execute cudaMemcpy(cudaMemcpyDeviceToHost)\n");
        exit(0);
    }
}

extern "C" void cuda_device_synchronize()
{
    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        printf("failed to execute cudaDeviceSynchronize()\n");
        exit(0);
    }
}
        

cudaStream_t* streams;

extern "C" void cuda_create_streams(int num_streams)
{
    streams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
    for (int i = 0; i < num_streams; i++) cudaStreamCreate(&streams[i]);
}

extern "C" void cuda_destroy_streams(int num_streams)
{
    for (int i = 0; i < num_streams; i++) cudaStreamDestroy(streams[i]);
    free(streams);
}

extern "C" void cuda_stream_synchronize(int stream_id)
{
    if (cudaStreamSynchronize(streams[stream_id]) != cudaSuccess)
    {
        printf("failed to execute cudaStreamSynchronize()\n");
        exit(0);
    }
}

extern "C" void cuda_async_copy_to_device(void* target, void* source, size_t size, int stream_id)
{
    cudaStream_t stream = (stream_id == -1) ? NULL : streams[stream_id];

    if (cudaMemcpyAsync(target, source, size, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        printf("failed to execute cudaMemcpy(cudaMemcpyHostToDevice)\n");
        exit(0);
    }
}

extern "C" void cuda_async_copy_to_host(void* target, void* source, size_t size, int stream_id)
{
    cudaStream_t stream = (stream_id == -1) ? NULL : streams[stream_id];

    if (cudaMemcpyAsync(target, source, size, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        printf("failed to execute cudaMemcpy(cudaMemcpyDeviceToHost)\n");
        exit(0);
    }
}

extern "C" void cuda_memset(void* ptr,int value, size_t size)
{
    if (cudaMemset(ptr, value, size) != cudaSuccess)
    {
        printf("failed to execute cudaMemset()\n");
        exit(0);
    }
}

extern "C" void cuda_host_register(void* ptr, size_t size)
{
    assert(ptr);
    
    cudaError_t err = cudaHostRegister(ptr, size, 0);
    if (err != cudaSuccess)
    {
        printf("failed to execute cudaHostRegister\n");
        switch (err)
        {
            case cudaErrorInvalidValue:
                printf("cudaErrorInvalidValue\n");
                break;
            case cudaErrorMemoryAllocation:
                printf("cudaErrorMemoryAllocation\n");
                break;
            default:
                printf("unrecognized error\n");
        }
        exit(-1);
    }
}

extern "C" void cuda_host_unregister(void* ptr)
{
    if (cudaHostUnregister(ptr) != cudaSuccess)
    {
        printf("failed to execute cudaHostUnregister\n");
        exit(-1);
    }
}

//* cudaDeviceProp& cuda_devprop()
//* {
//*     static cudaDeviceProp devprop;
//* 
//*     return devprop;
//* }

extern "C" size_t cuda_get_free_mem()
{
    size_t free, total;
    
    cudaMemGetInfo(&free, &total);

    return free;
}

extern "C" void cuda_device_info()
{
    int count;
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {
        printf("failed to execute cudaGetDeviceCount() \n");
        exit(-1);
    }

    if (count == 0)
    {
        printf("no avaiable devices\n");
        exit(-1);
    }

    cudaDeviceProp devprop;
     
    if (cudaGetDeviceProperties(&devprop, 0) != cudaSuccess)
    {
        printf("failed to execute cudaGetDeviceProperties()\n");
        exit(-1);
    }
    
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

//==================
// CUBLAS functions
//==================

cublasHandle_t& cublas_handle()
{
    static cublasHandle_t handle;
    static bool init = false;

    if (!init)
    {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
        {
            printf("cublasCreate() failed \n");
            exit(-1);
        }
        init = true;
    }
    
    return handle;
}

extern "C" void cublas_init()
{
    cublas_handle();
}

extern "C" void cublas_zgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                             void* alpha, void* a, int32_t lda, void* b, 
                             int32_t ldb, void* beta, void* c, int32_t ldc)
{
    const cublasOperation_t trans[] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};
    
    
    cublasStatus_t status = cublasZgemm(cublas_handle(), trans[transa], trans[transb], m, n, k, (cuDoubleComplex*)alpha, 
                                        (cuDoubleComplex*)a, lda, (cuDoubleComplex*)b, ldb, (cuDoubleComplex*)beta, 
                                        (cuDoubleComplex*)c, ldc);
    if (status == CUBLAS_STATUS_SUCCESS) return;

    printf("failed to execute cublasZgemm\n");
    
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
            printf("he device does not support double-precision\n");
            break;
        }
        case CUBLAS_STATUS_EXECUTION_FAILED:
        {
            printf("the function failed to launch on the GPU\n");
            break;
        }
    }

    exit(-1);
}

// A(GPU) => B(CPU)
extern "C" void cublas_get_matrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb)
{
    if (cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb) != CUBLAS_STATUS_SUCCESS)
    {
        printf("failed to execute cublasGetMatrix\n");
        exit(-1);
    }
}

// A(CPU) => B(GPU)
extern "C" void cublas_set_matrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb)
{
    if (cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb) != CUBLAS_STATUS_SUCCESS)
    {
        printf("failed to execute cublasSetMatrix\n");
        exit(-1);
    }
}

//=================
// CUFFT functions
//=================

cufftHandle plan;
int nfft_of_plan;
int size_of_plan;

extern "C" void cufft_create_batch_plan(int nx, int ny, int nz, int nfft)
{
    int fft_size = nx * ny * nz;
    int n[] = {nz, ny, nx};

    cufftResult result = cufftPlanMany(&plan, 3, n, n, 1, fft_size, n, 1, fft_size, CUFFT_Z2Z, nfft);
    if (result != CUFFT_SUCCESS)
    {
        printf("failed to execute cufftPlanMany()\n");
        exit(0);
    }

    nfft_of_plan = nfft;
    size_of_plan = fft_size;
}

extern "C" void cufft_destroy_batch_plan()
{
    cufftDestroy(plan);
}

//== __global__ void cufft_batch_load_kernel(int fft_size, int num_gkvec, int* map, cuDoubleComplex* phi, 
//==                                         cuDoubleComplex* fft_buffer)
//== {
//==     int i = blockIdx.y;
//==     int ig = blockDim.x * blockIdx.x + threadIdx.x;
//== 
//==     if (ig < num_gkvec) fft_buffer[array2D_offset(map[ig], i, fft_size)] = phi[array2D_offset(ig, i, num_gkvec)];
//== }

//= __global__ void cufft_batch_apply_v_kernel(int fft_size, cuDoubleComplex* v_r, cuDoubleComplex* fft_buffer)
//= {
//=     int i = blockIdx.y;
//=     int ir = blockDim.x * blockIdx.x + threadIdx.x;
//=     if (ir < fft_size) 
//=     {
//=         fft_buffer[array2D_offset(ir, i, fft_size)] = 
//=             cuCmul(fft_buffer[array2D_offset(ir, i, fft_size)], v_r[ir]);
//=     }
//= }

//== __global__ void cufft_batch_unload_kernel(int fft_size, int num_gkvec, int* map, cuDoubleComplex* fft_buffer,
//==                                           cuDoubleComplex* phi)
//== {
//==     int i = blockIdx.y;
//==     int ig = blockDim.x * blockIdx.x + threadIdx.x;
//== 
//==     if (ig < num_gkvec) 
//==     {
//==         phi[array2D_offset(ig, i, num_gkvec)] = 
//==             cuCdiv(fft_buffer[array2D_offset(map[ig], i, fft_size)], make_cuDoubleComplex(double(fft_size), 0));
//==     }
//== }

//== extern "C" void cufft_batch_apply_v(int fft_size, int num_gkvec, int num_phi, void* buffer, int* map, void* v_r, void* p)
//== {
//==     dim3 threadsPerBlock(64);
//==     dim3 numBlocks(num_blocks(num_gkvec, 64), num_phi);
//==     
//==     cuda_memset(buffer, 0, fft_size * num_phi * sizeof(cuDoubleComplex));
//== 
//==     cufft_batch_load_kernel<<<numBlocks, threadsPerBlock>>>
//==         (fft_size, num_gkvec, map, (cuDoubleComplex*)p, (cuDoubleComplex*)buffer);
//==     
//==     cufftExecZ2Z(plan, (cufftDoubleComplex*)buffer, (cufftDoubleComplex*)buffer, CUFFT_INVERSE);
//==     
//==     //dim3 numBlocks_r(num_blocks(fft_size, 64), num_phi);
//==     //cufft_batch_apply_v_kernel<<<numBlocks_r, threadsPerBlock>>>
//==     //    (fft_size, (cuDoubleComplex*)v_r, (cuDoubleComplex*)buffer);
//==     
//==     cufftExecZ2Z(plan, (cufftDoubleComplex*)buffer, (cufftDoubleComplex*)buffer, CUFFT_FORWARD);
//== 
//==     cufft_batch_unload_kernel<<<numBlocks, threadsPerBlock>>>
//==         (fft_size, num_gkvec, map, (cuDoubleComplex*)buffer, (cuDoubleComplex*)p);
//== }



__global__ void cufft_normalize(int size, cuDoubleComplex* buffer)
{
    int i = blockIdx.y;
    int ir = blockDim.x * blockIdx.x + threadIdx.x;

    if (ir < size) 
    {
        buffer[array2D_offset(ir, i, size)] = 
            cuCdiv(buffer[array2D_offset(ir, i, size)], make_cuDoubleComplex(double(size), 0));
    }
}

extern "C" void cufft_forward_transform(void* buffer)
{
    cufftExecZ2Z(plan, (cufftDoubleComplex*)buffer, (cufftDoubleComplex*)buffer, CUFFT_FORWARD);
    
    dim3 threadsPerBlock(64);
    dim3 numBlocks(num_blocks(size_of_plan, 64), nfft_of_plan);
    cufft_normalize<<<numBlocks, threadsPerBlock>>>(size_of_plan, (cufftDoubleComplex*)buffer);
}

extern "C" void cufft_backward_transform(void* buffer)
{
    cufftExecZ2Z(plan, (cufftDoubleComplex*)buffer, (cufftDoubleComplex*)buffer, CUFFT_INVERSE);
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
    


__global__ void scale_matrix_columns_gpu_kernel(int nrow, cuDoubleComplex* mtrx, double* a)
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
extern "C" void scale_matrix_columns_gpu(int nrow, int ncol, void* mtrx, double* a)
{
    dim3 threadsPerBlock(64);
    dim3 numBlocks(num_blocks(nrow, 64), ncol);
    scale_matrix_columns_gpu_kernel<<<numBlocks, threadsPerBlock>>>(nrow, (cuDoubleComplex*)mtrx, a);
}

__global__ void scale_matrix_rows_gpu_kernel(int nrow, cuDoubleComplex* mtrx, double* v)
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
extern "C" void scale_matrix_rows_gpu(int nrow, int ncol, void* mtrx, double* v)
{
    dim3 threadsPerBlock(64);
    dim3 numBlocks(num_blocks(nrow, 64), ncol);

    scale_matrix_rows_gpu_kernel<<<numBlocks, threadsPerBlock>>>(nrow, (cuDoubleComplex*)mtrx, v);
}

__global__ void create_beta_pw_gpu_kernel(int num_gkvec, 
                                          int* beta_t_idx, 
                                          cuDoubleComplex* beta_pw_type, 
                                          double* gkvec, 
                                          double* atom_pos,
                                          cuDoubleComplex* beta_pw)
{
    const double twopi = 6.2831853071795864769;

    int i = blockIdx.y;
    int ia = beta_t_idx[array2D_offset(0, i, 2)];
    int offset_t = beta_t_idx[array2D_offset(1, i, 2)];

    int igk = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (igk < num_gkvec)
    {
        double p = 0;
        for (int x = 0; x < 3; x++) p += atom_pos[array2D_offset(x, ia, 3)] * gkvec[array2D_offset(x, igk, 3)];
        p *= twopi;
        
        double sinp = sin(p);
        double cosp = cos(p);

        beta_pw[array2D_offset(igk, i, num_gkvec)] = 
            cuCmul(beta_pw_type[array2D_offset(igk, offset_t, num_gkvec)], make_cuDoubleComplex(cosp, -sinp));
    }
}

extern "C" void create_beta_pw_gpu(int num_gkvec, 
                                   int num_beta_atot, 
                                   int* beta_t_idx,
                                   void* beta_pw_type,
                                   double* gkvec,
                                   double* atom_pos,
                                   void* beta_pw)
{
    dim3 threadsPerBlock(64);
    dim3 numBlocks(num_blocks(num_gkvec, 64), num_beta_atot);

    create_beta_pw_gpu_kernel<<<numBlocks, threadsPerBlock>>>(num_gkvec, 
                                                              beta_t_idx, 
                                                              (cuDoubleComplex*)beta_pw_type,
                                                              gkvec,
                                                              atom_pos,
                                                              (cuDoubleComplex*)beta_pw);
}


