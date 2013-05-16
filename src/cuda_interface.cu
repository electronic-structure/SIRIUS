// This file must be compiled with nvcc

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cublas_v2.h>

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

/*extern "C" void cuda_init()
{
    if (cuInit(0) != CUDA_SUCCESS)
    {
        printf("cuInit failed\n");
    }
}*/

extern "C" void cublas_init()
{
    cublas_handle();
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

extern "C" void cuda_malloc(void **ptr, size_t size)
{
    if (cudaMalloc(ptr, size) != cudaSuccess)
    {
        printf("failed to execute cudaMalloc() \n");
        exit(0);
    }
}

extern "C" void cuda_free(void *ptr)
{
    if (cudaFree(ptr) != cudaSuccess)
    {
        printf("failed to execute cudaFree() \n");
        exit(0);
    }
}

extern "C" void cuda_copy_to_device(void *target, void *source, size_t size)
{
    if (cudaMemcpy(target, source, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        printf("failed to execute cudaMemcpy(cudaMemcpyHostToDevice)\n");
        exit(0);
    }
}

extern "C" void cuda_copy_to_host(void *target, void *source, size_t size)
{
    if (cudaMemcpy(target, source, size, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        printf("failed to execute cudaMemcpy(cudaMemcpyDeviceToHost)\n");
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

extern "C" void cuda_async_copy_to_device(void *target, void *source, size_t size, int stream_id)
{
    cudaStream_t stream = (stream_id == -1) ? NULL : streams[stream_id];

    if (cudaMemcpyAsync(target, source, size, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        printf("failed to execute cudaMemcpy(cudaMemcpyHostToDevice)\n");
        exit(0);
    }
}

extern "C" void cuda_async_copy_to_host(void *target, void *source, size_t size, int stream_id)
{
    cudaStream_t stream = (stream_id == -1) ? NULL : streams[stream_id];

    if (cudaMemcpyAsync(target, source, size, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        printf("failed to execute cudaMemcpy(cudaMemcpyDeviceToHost)\n");
        exit(0);
    }
}

extern "C" void cuda_memset(void *ptr,int value, size_t size)
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
}


extern "C" void cublas_zgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                             void* alpha, void* a, int32_t lda, void* b, 
                             int32_t ldb, void* beta, void* c, int32_t ldc)
{
    const cublasOperation_t trans[] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};

    if (cublasZgemm(cublas_handle(), trans[transa], trans[transb], m, n, k, (cuDoubleComplex*)alpha, (cuDoubleComplex*)a, lda, 
                   (cuDoubleComplex*)b, ldb, (cuDoubleComplex*)beta, (cuDoubleComplex*)c, ldc) != CUBLAS_STATUS_SUCCESS)
    {
        printf("failed to execute cublasZgemm() \n");
        exit(-1);
    }
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


__device__ size_t array2D_offset(int i0, int i1, int ld0)
{
    return i0 + i1 * ld0;
}

__device__ size_t array3D_offset(int i0, int i1, int i2, int ld0, int ld1)
{
    return i0 + i1 * ld0 + i2 * ld0 * ld1;
}

__device__ size_t array4D_offset(int i0, int i1, int i2, int i3, int ld0, int ld1, int ld2)
{
    return i0 + i1 * ld0 + i2 * ld0 * ld1 + i3 * ld0 * ld1 * ld2;
}

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

    //for (int s = 1; s < blockDim.x; s *= 2) 
    //{
    //    if (threadIdx.x % (2 * s) == 0) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    //    __syncthreads();
    //}
    
    if (threadIdx.x == 0) for (int i = 1; i < blockDim.x; i++) sdata[0] += sdata[i];

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

    ////for (int s = 1; s < blockDim.x; s *= 2) 
    ////{
    ////    if (threadIdx.x % (2 * s) == 0) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    ////    __syncthreads();
    ////}
    //
    if (threadIdx.x == 0) for (int i = 1; i < blockDim.x; i++) sdata[0] = cuCadd(sdata[0], sdata[i]);

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




__global__ void bessel_lo_inner_product_gpu_kernel(int max_num_mt_points, int lmax_pw, int num_atom_types, int num_gkvec, 
                                                   double* jl_coefs, double* lo_coefs, int* l_by_ilo, int* iat_by_ilo, 
                                                   int* nmtp_by_ilo, double* r_dr, double* jlo)
{
    int igk = blockIdx.x;
    int ilo = blockIdx.y;
    int l = l_by_ilo[ilo];
    int iat = iat_by_ilo[ilo];
    int nmtp = nmtp_by_ilo[ilo];

    int ld1 = max_num_mt_points * 4;
    int ld2 = ld1 * (lmax_pw + 1);
    int ld3 = ld2 * num_atom_types;
    double* jl_ptr = &jl_coefs[l * ld1 + iat * ld2 + igk * ld3];
    double* lo_ptr = &lo_coefs[ld1 * ilo];
    double* r_dr_ptr = &r_dr[2 * max_num_mt_points * iat];
    
    jlo[igk + ilo * num_gkvec] = spline_inner_product_gpu_function(max_num_mt_points, nmtp, r_dr_ptr, jl_ptr, lo_ptr);
}


void bessel_lo_inner_product_gpu(int num_gkvec, int num_lo, int max_num_mt_points, int lmax_pw, int num_atom_types, 
                                 double* jl_coefs, double* lo_coefs, int* l_by_ilo, int* iat_by_ilo, int* nmtp_by_ilo, 
                                 double* r_dr, double* jlo)
{
    dim3 threadsPerBlock(64);
    dim3 numBlocks(num_gkvec, num_lo);

    bessel_lo_inner_product_gpu_kernel<<<numBlocks, threadsPerBlock, 64 * 16>>>
        (max_num_mt_points, lmax_pw, num_atom_types, num_gkvec, jl_coefs, lo_coefs, l_by_ilo, iat_by_ilo, nmtp_by_ilo,
         r_dr, jlo);
}

__global__ void bessel_vlo_inner_product_gpu_kernel(int max_num_mt_points, int lmax_pw, int lmmax_pw, int num_atom_types, 
                                                    int num_gkvec, double* jl_coefs, cuDoubleComplex* vlo_coefs,
                                                    int* l_by_lm, int* iat_by_ilo, int* nmtp_by_ilo, double* r_dr, 
                                                    cuDoubleComplex* jvlo)

{
    int igk = blockIdx.x;
    int ilo = blockIdx.y;
    int lm = blockIdx.z;
    int l = l_by_lm[lm];
    int iat = iat_by_ilo[ilo];
    int nmtp = nmtp_by_ilo[ilo];
    
    int ld1 = max_num_mt_points * 4;
    int ld2 = ld1 * (lmax_pw + 1);
    int ld3 = ld2 * num_atom_types;
    double* jl_ptr = &jl_coefs[l * ld1 + iat * ld2 + igk * ld3];
    cuDoubleComplex* vlo_ptr = &vlo_coefs[ld1 * lm + ld1 * lmmax_pw * ilo];
    double* r_dr_ptr = &r_dr[2 * max_num_mt_points * iat];
    
    jvlo[lm + igk * lmmax_pw + ilo * lmmax_pw * num_gkvec] = 
        spline_inner_product_gpu_function(max_num_mt_points, nmtp, r_dr_ptr, jl_ptr, vlo_ptr);
}



void bessel_vlo_inner_product_gpu(int num_gkvec, int num_lo, int max_num_mt_points, int lmax_pw, int lmmax_pw, int num_atom_types, 
                                 double* jl_coefs, void* vlo_coefs, int* l_by_lm, int* iat_by_ilo, int* nmtp_by_ilo, 
                                 double* r_dr, void* jvlo)
{
    dim3 threadsPerBlock(128);
    dim3 numBlocks(num_gkvec, num_lo, lmmax_pw);

    bessel_vlo_inner_product_gpu_kernel<<<numBlocks, threadsPerBlock, 128 * 16>>>
        (max_num_mt_points, lmax_pw, lmmax_pw, num_atom_types, num_gkvec, jl_coefs, (cuDoubleComplex*)vlo_coefs, 
         l_by_lm, iat_by_ilo, nmtp_by_ilo, r_dr, (cuDoubleComplex*)jvlo);
}

__global__ void bessel_vlm_inner_product_gpu_kernel(int max_num_mt_points, int lmax_pot, int lmmax_pot, int* iat_by_ia,
                                                    int* nmtp_by_ia, int* l_by_lm, double* r_dr, double* bessel, 
                                                    double* vlm, double* vjlm)
{
    int lm = blockIdx.x;
    int ia = blockIdx.y;
    int iat = iat_by_ia[ia];
    int nmtp = nmtp_by_ia[ia];
    int l = l_by_lm[lm];

    int ld1 = max_num_mt_points * 4;
    
    double* jl_ptr = &bessel[l * ld1 + iat * ld1 * (lmax_pot + 1)];
    double* vlm_ptr = &vlm[lm * ld1 + ia * ld1 * lmmax_pot];
    double* r_dr_ptr = &r_dr[2 * max_num_mt_points * iat];

    vjlm[lm + ia * lmmax_pot] = spline_inner_product_gpu_function(max_num_mt_points, nmtp, r_dr_ptr, jl_ptr, vlm_ptr);
}


void bessel_vlm_inner_product_gpu(int max_num_mt_points, int lmax_pot, int lmmax_pot, int num_atoms, int num_atom_types, 
                                  int* iat_by_ia, int* nmtp_by_ia, int* l_by_lm, double* r_dr, double* jl_coefs, 
                                  double* vlm_coefs, double* vjlm, int stream_id)
{
    cudaStream_t stream = (stream_id == -1) ? NULL : streams[stream_id];
    dim3 threadsPerBlock(64);
    dim3 numBlocks(lmmax_pot, num_atoms);
    
    bessel_vlm_inner_product_gpu_kernel<<<numBlocks, threadsPerBlock, 64 * 16, stream>>>
        (max_num_mt_points, lmax_pot, lmmax_pot, iat_by_ia, nmtp_by_ia, l_by_lm, r_dr, jl_coefs, vlm_coefs, vjlm);
}









