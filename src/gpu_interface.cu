#include <stdio.h>
#include <assert.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <magmablas.h>
#include <magma_lapack.h>
#include <magma.h>

#include "typedefs.h"

cudaDeviceProp& cuda_devprop()
{
    static cudaDeviceProp devprop;

    return devprop;

}

cublasHandle_t& cublas_handle()
{
    static cublasHandle_t handle;
    static bool init = false;

    if (!init)
    {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
        {
            printf("cublasCreate() failed \n");
            exit(0);
        }
        init = true;
    }
    
    return handle;
}

extern "C" void init_gpu()
{
    int count;
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {
        printf("init_gpu: failed to execute cudaGetDeviceCount() \n");
        return;
    }

    if (count == 0)
    {
        printf("init_gpu: no avaiable devices\n");
    }

    cudaDeviceProp devprop;
     
    if (cudaGetDeviceProperties(&devprop, 0) != cudaSuccess)
    {
        printf("init_gpu: failed to execute cudaGetDeviceProperties()\n");
        return;
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
    printf("memPitch                    : %i \n", devprop.memPitch);
    printf("multiProcessorCount         : %i \n", devprop.multiProcessorCount);
    printf("regsPerBlock                : %i \n", devprop.regsPerBlock);
    printf("sharedMemPerBlock           : %i kB \n", devprop.sharedMemPerBlock/1024);
    printf("totalConstMem               : %i kB \n", devprop.totalConstMem/1024);
    printf("totalGlobalMem              : %i kB \n", devprop.totalGlobalMem/1024);
}

extern "C" void gpu_malloc(void **ptr, int size)
{
    if (cudaMalloc(ptr, size) != cudaSuccess)
    {
        printf("failed to execute cudaMalloc() \n");
        exit(0);
    }
}

extern "C" void gpu_free(void *ptr)
{
    if (cudaFree(ptr) != cudaSuccess)
    {
        printf("failed to execute cudaFree() \n");
        exit(0);
    }
}

extern "C" void gpu_copy_to_device(void *target, void *source, int size)
{
    if (cudaMemcpy(target, source, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        printf("failed to execute cudaMemcpy(cudaMemcpyHostToDevice)\n");
        exit(0);
    }
}

extern "C" void gpu_copy_to_host(void *target, void *source, int size)
{
    if (cudaMemcpy(target, source, size, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        printf("failed to execute cudaMemcpy(cudaMemcpyDeviceToHost)\n");
        exit(0);
    }
}

extern "C" void gpu_mem_zero(void *ptr, int size)
{
    if (cudaMemset(ptr, 0, size) != cudaSuccess)
    {
        printf("failed to execute cudaMemset()\n");
        exit(0);
    }
}

extern "C" void gpu_zgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                          complex16 alpha, complex16 *a, int32_t lda, complex16 *b, 
                          int32_t ldb, complex16 beta, complex16 *c, int32_t ldc)
{
    assert(sizeof(cuDoubleComplex) == sizeof(complex16));
    
    const cublasOperation_t trans[] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};

    if (cublasZgemm(cublas_handle(), trans[transa], trans[transb], m, n, k, (cuDoubleComplex *)(&alpha), (cuDoubleComplex *)a, lda, 
                    (cuDoubleComplex *)b, ldb, (cuDoubleComplex *)(&beta), (cuDoubleComplex *)c, ldc) != CUBLAS_STATUS_SUCCESS)
    {
        printf("failed to execute cublasZgemm() \n");
        exit(0);
    }
}

extern "C" void gpu_zhegvx(int32_t n, int32_t nv, double abstol, void *a, void *b,
                           double *eval, void *z, int32_t ldz)
{
    magma_int_t m1, info;

    magma_int_t nb = magma_get_zhetrd_nb(n);
    magma_int_t lwork = 2 * n * (nb + 1);
    magma_int_t lrwork = 7 * n;
    magma_int_t liwork = 6 * n;
    
    cuDoubleComplex *h_work;
    double *rwork, *w1;
    magma_int_t *iwork, *ifail;
    
    w1 = (double *)malloc(n * sizeof(double));
    h_work = (cuDoubleComplex *)malloc(lwork * sizeof(cuDoubleComplex));
    rwork = (double *)malloc(lrwork * sizeof(double));
    iwork = (magma_int_t *)malloc(liwork * sizeof(magma_int_t));
    ifail = iwork + 5 * n;

    magma_zhegvx(1, 'V', 'I', 'U', n, (cuDoubleComplex *)a, n, (cuDoubleComplex *)b, n, 0.0, 0.0, 1, nv, abstol, 
                 &m1, w1, (cuDoubleComplex *)z, ldz, h_work, lwork, rwork, iwork, ifail, &info);

    memcpy(eval, &w1[0], nv * sizeof(double)); 
    
    free(iwork);
    free(rwork);
    free(w1);
    free(h_work);
}
 
