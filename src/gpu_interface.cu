// This file must be compiled with nvcc

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <magmablas.h>
#include <magma_lapack.h>
#include <magma.h>

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

extern "C" void cuda_init()
{
    if (cuInit(0) != CUDA_SUCCESS)
    {
        printf("cuInit failed\n");
    }
}

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

extern "C" magma_int_t magma_zbulge_get_lq2(magma_int_t n);

extern "C" magma_int_t magma_zhegvdx_2stage(magma_int_t itype, char jobz, char range, char uplo, magma_int_t n,
                                            cuDoubleComplex *a, magma_int_t lda, cuDoubleComplex *b, magma_int_t ldb,
                                            double vl, double vu, magma_int_t il, magma_int_t iu,
                                            magma_int_t *m, double *w, cuDoubleComplex *work, magma_int_t lwork,
                                            double *rwork, magma_int_t lrwork,
                                            magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);

extern "C" void magma_zhegvdx_2stage_wrapper(int32_t matrix_size, int32_t nv, void* a, int32_t lda, void* b, 
                                             int32_t ldb, double* eval)
{
    int m;
    int info;

    int lwork = magma_zbulge_get_lq2(matrix_size) + 2 * matrix_size + matrix_size * matrix_size;
    int lrwork = 1 + 5 * matrix_size + 2 * matrix_size * matrix_size;
    int liwork = 3 + 5 * matrix_size;
            
    cuDoubleComplex* h_work;
    cuda_malloc_host((void**)&h_work, lwork * sizeof(cuDoubleComplex));

    double* rwork;
    cuda_malloc_host((void**)&rwork, lrwork * sizeof(double));
    
    magma_int_t *iwork;
    if ((iwork = (magma_int_t*)malloc(liwork * sizeof(magma_int_t))) == NULL)
    {
        printf("malloc failed\n");
        exit(-1);
    }
    
    double* w;
    if ((w = (double*)malloc(matrix_size * sizeof(double))) == NULL)
    {
        printf("mallof failed\n");
        exit(-1);
    }

    magma_zhegvdx_2stage(1, 'V', 'I', 'L', matrix_size, (cuDoubleComplex*)a, lda, (cuDoubleComplex*)b, ldb, 0.0, 0.0, 
                         1, nv, &m, w, h_work, lwork, rwork, lrwork, iwork, liwork, &info);

    memcpy(eval, &w[0], nv * sizeof(double));
    
    cuda_free_host((void**)&h_work);
    cuda_free_host((void**)&rwork);
    free(iwork);
    free(w);

    if (m != nv)
    {
        printf("Not all eigen-values are found.\n");
        exit(-1);
    }
    
    if (info)
    {
        printf("magma_zhegvdx_2stage returned : %i\n", info);
        exit(-1);
    }
}



//* cudaDeviceProp& cuda_devprop()
//* {
//*     static cudaDeviceProp devprop;
//* 
//*     return devprop;
//* 
//* }
//* 
//* 
//* extern "C" void init_gpu()
//* {
//*     int count;
//*     if (cudaGetDeviceCount(&count) != cudaSuccess)
//*     {
//*         printf("init_gpu: failed to execute cudaGetDeviceCount() \n");
//*         return;
//*     }
//* 
//*     if (count == 0)
//*     {
//*         printf("init_gpu: no avaiable devices\n");
//*     }
//* 
//*     cudaDeviceProp devprop;
//*      
//*     if (cudaGetDeviceProperties(&devprop, 0) != cudaSuccess)
//*     {
//*         printf("init_gpu: failed to execute cudaGetDeviceProperties()\n");
//*         return;
//*     }
//*     
//*     printf("name                        : %s \n", devprop.name);
//*     printf("major                       : %i \n", devprop.major);
//*     printf("minor                       : %i \n", devprop.minor);
//*     printf("asyncEngineCount            : %i \n", devprop.asyncEngineCount);
//*     printf("canMapHostMemory            : %i \n", devprop.canMapHostMemory);
//*     printf("clockRate                   : %i kHz \n", devprop.clockRate);
//*     printf("concurrentKernels           : %i \n", devprop.concurrentKernels);
//*     printf("ECCEnabled                  : %i \n", devprop.ECCEnabled);
//*     printf("l2CacheSize                 : %i kB \n", devprop.l2CacheSize/1024);
//*     printf("maxGridSize                 : %i %i %i \n", devprop.maxGridSize[0], devprop.maxGridSize[1], devprop.maxGridSize[2]);
//*     printf("maxThreadsDim               : %i %i %i \n", devprop.maxThreadsDim[0], devprop.maxThreadsDim[1], devprop.maxThreadsDim[2]);
//*     printf("maxThreadsPerBlock          : %i \n", devprop.maxThreadsPerBlock);
//*     printf("maxThreadsPerMultiProcessor : %i \n", devprop.maxThreadsPerMultiProcessor);
//*     printf("memoryBusWidth              : %i bits \n", devprop.memoryBusWidth);
//*     printf("memoryClockRate             : %i kHz \n", devprop.memoryClockRate);
//*     printf("memPitch                    : %i \n", devprop.memPitch);
//*     printf("multiProcessorCount         : %i \n", devprop.multiProcessorCount);
//*     printf("regsPerBlock                : %i \n", devprop.regsPerBlock);
//*     printf("sharedMemPerBlock           : %i kB \n", devprop.sharedMemPerBlock/1024);
//*     printf("totalConstMem               : %i kB \n", devprop.totalConstMem/1024);
//*     printf("totalGlobalMem              : %i kB \n", devprop.totalGlobalMem/1024);
//* }
//* 

extern "C" void cuda_malloc(void **ptr, int size)
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

extern "C" void cuda_copy_to_device(void *target, void *source, int size)
{
    if (cudaMemcpy(target, source, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        printf("failed to execute cudaMemcpy(cudaMemcpyHostToDevice)\n");
        exit(0);
    }
}

extern "C" void cuda_copy_to_host(void *target, void *source, int size)
{
    if (cudaMemcpy(target, source, size, cudaMemcpyDeviceToHost) != cudaSuccess)
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

//* extern "C" void gpu_zgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
//*                           complex16 alpha, complex16 *a, int32_t lda, complex16 *b, 
//*                           int32_t ldb, complex16 beta, complex16 *c, int32_t ldc)
//* {
//*     assert(sizeof(cuDoubleComplex) == sizeof(complex16));
//*     
//*     const cublasOperation_t trans[] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};
//* 
//*     if (cublasZgemm(cublas_handle(), trans[transa], trans[transb], m, n, k, (cuDoubleComplex *)(&alpha), (cuDoubleComplex *)a, lda, 
//*                     (cuDoubleComplex *)b, ldb, (cuDoubleComplex *)(&beta), (cuDoubleComplex *)c, ldc) != CUBLAS_STATUS_SUCCESS)
//*     {
//*         printf("failed to execute cublasZgemm() \n");
//*         exit(0);
//*     }
//* }
//* 
//* extern "C" void gpu_zhegvx(int32_t n, int32_t nv, double abstol, void *a, void *b,
//*                            double *eval, void *z, int32_t ldz)
//* {
//*     magma_int_t m1, info;
//* 
//*     magma_int_t nb = magma_get_zhetrd_nb(n);
//*     magma_int_t lwork = 2 * n * (nb + 1);
//*     magma_int_t lrwork = 7 * n;
//*     magma_int_t liwork = 6 * n;
//*     
//*     cuDoubleComplex *h_work;
//*     double *rwork, *w1;
//*     magma_int_t *iwork, *ifail;
//*     
//*     w1 = (double *)malloc(n * sizeof(double));
//*     h_work = (cuDoubleComplex *)malloc(lwork * sizeof(cuDoubleComplex));
//*     rwork = (double *)malloc(lrwork * sizeof(double));
//*     iwork = (magma_int_t *)malloc(liwork * sizeof(magma_int_t));
//*     ifail = iwork + 5 * n;
//* 
//*     magma_zhegvx(1, 'V', 'I', 'U', n, (cuDoubleComplex *)a, n, (cuDoubleComplex *)b, n, 0.0, 0.0, 1, nv, abstol, 
//*                  &m1, w1, (cuDoubleComplex *)z, ldz, h_work, lwork, rwork, iwork, ifail, &info);
//* 
//*     memcpy(eval, &w1[0], nv * sizeof(double)); 
//*     
//*     free(iwork);
//*     free(rwork);
//*     free(w1);
//*     free(h_work);
//* }
//*  
