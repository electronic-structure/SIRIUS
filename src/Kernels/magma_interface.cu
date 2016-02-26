// This file must be compiled with nvcc

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <magma.h>
#include <magma_z.h>
#include <magma_threadsetting.h>

extern "C" void magma_init_wrapper()
{
    magma_init();
}

extern "C" void magma_finalize_wrapper()
{
    magma_finalize();
}

extern "C" void magma_zhegvdx_2stage_wrapper(int32_t matrix_size, int32_t nv, void* a, int32_t lda, void* b, 
                                             int32_t ldb, double* eval)
{
    int m;
    int info;
    
    int lwork;
    int lrwork;
    int liwork;
    magma_zheevdx_getworksize(matrix_size, magma_get_parallel_numthreads(), 1, &lwork, &lrwork, &liwork);

    //int lwork = magma_get_zbulge_lq2(matrix_size, magma_get_parallel_numthreads(), 1) + 2 * matrix_size + matrix_size * matrix_size;
    //int lrwork = 1 + 5 * matrix_size + 2 * matrix_size * matrix_size;
    //int liwork = 3 + 5 * matrix_size;
            
    magmaDoubleComplex* h_work;
    if (cudaMallocHost((void**)&h_work, lwork * sizeof(magmaDoubleComplex)) != cudaSuccess)
    {
        printf("cudaMallocHost failed at line %i of file %s\n", __LINE__, __FILE__);
        exit(-1);
    }
    double* rwork;
    if (cudaMallocHost((void**)&rwork, lrwork * sizeof(double)) != cudaSuccess)
    {
        printf("cudaMallocHost failed at line %i of file %s\n", __LINE__, __FILE__);
        exit(-1);
    }
    
    magma_int_t *iwork;
    if ((iwork = (magma_int_t*)malloc(liwork * sizeof(magma_int_t))) == NULL)
    {
        printf("malloc failed\n");
        exit(-1);
    }
    
    double* w;
    if ((w = (double*)malloc(matrix_size * sizeof(double))) == NULL)
    {
        printf("malloc failed\n");
        exit(-1);
    }

    magma_zhegvdx_2stage(1, MagmaVec, MagmaRangeI, MagmaLower, matrix_size, (magmaDoubleComplex*)a, lda, (magmaDoubleComplex*)b, ldb, 0.0, 0.0, 
                         1, nv, &m, w, h_work, lwork, rwork, lrwork, iwork, liwork, &info);

    if (info)
    {
        printf("magma_zhegvdx_2stage returned : %i\n", info);
        if (info == MAGMA_ERR_DEVICE_ALLOC)
            printf("this is MAGMA_ERR_DEVICE_ALLOC\n");
        exit(-1);
    }    

    if (m < nv)
    {
        printf("Not all eigen-vectors are found.\n");
        printf("requested number of eigen-vectors: %i\n", nv);
        printf("found number of eigen-vectors: %i\n", m);
        exit(-1);
    }

    memcpy(eval, &w[0], nv * sizeof(double));
    
    cudaFreeHost(h_work);
    cudaFreeHost(rwork);
    free(iwork);
    free(w);

}

extern "C" void magma_dsygvdx_2stage_wrapper(int32_t matrix_size, int32_t nv, void* a, int32_t lda, void* b, 
                                             int32_t ldb, double* eval)
{
    int m;
    int info;
    
    int lwork;
    //int lrwork;
    int liwork;
    magma_dsyevdx_getworksize(matrix_size, magma_get_parallel_numthreads(), 1, &lwork, &liwork);

    double* h_work;
    if (cudaMallocHost((void**)&h_work, lwork * sizeof(double)) != cudaSuccess)
    {
        printf("cudaMallocHost failed at line %i of file %s\n", __LINE__, __FILE__);
        exit(-1);
    }
    //double* rwork;
    //if (cudaMallocHost((void**)&rwork, lrwork * sizeof(double)) != cudaSuccess)
    //{
    //    printf("cudaMallocHost failed at line %i of file %s\n", __LINE__, __FILE__);
    //    exit(-1);
    //}
    
    magma_int_t *iwork;
    if ((iwork = (magma_int_t*)malloc(liwork * sizeof(magma_int_t))) == NULL)
    {
        printf("malloc failed\n");
        exit(-1);
    }
    
    double* w;
    if ((w = (double*)malloc(matrix_size * sizeof(double))) == NULL)
    {
        printf("malloc failed\n");
        exit(-1);
    }

    magma_dsygvdx_2stage(1, MagmaVec, MagmaRangeI, MagmaLower, matrix_size, (double*)a, lda, (double*)b, ldb, 0.0, 0.0, 
                         1, nv, &m, w, h_work, lwork, iwork, liwork, &info);

    if (info)
    {
        printf("magma_dsygvdx_2stage : %i\n", info);
        if (info == MAGMA_ERR_DEVICE_ALLOC)
            printf("this is MAGMA_ERR_DEVICE_ALLOC\n");
        exit(-1);
    }    

    if (m < nv)
    {
        printf("Not all eigen-vectors are found.\n");
        printf("requested number of eigen-vectors: %i\n", nv);
        printf("found number of eigen-vectors: %i\n", m);
        exit(-1);
    }

    memcpy(eval, &w[0], nv * sizeof(double));
    
    cudaFreeHost(h_work);
    //cudaFreeHost(rwork);
    free(iwork);
    free(w);
}


