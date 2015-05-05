// This file must be compiled with nvcc

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <magma.h>
#include <magma_z.h>
#include <magma_zbulge.h>
#include <magma_threadsetting.h>
#include "gpu_interface.h"

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

    int lwork = magma_zbulge_get_lq2(matrix_size, magma_get_parallel_numthreads()) + 2 * matrix_size + matrix_size * matrix_size;
    int lrwork = 1 + 5 * matrix_size + 2 * matrix_size * matrix_size;
    int liwork = 3 + 5 * matrix_size;
            
    magmaDoubleComplex* h_work = (magmaDoubleComplex*)cuda_malloc_host(lwork * sizeof(magmaDoubleComplex));
    double* rwork = (double*)cuda_malloc_host(lrwork * sizeof(double));
    
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

    memcpy(eval, &w[0], nv * sizeof(double));
    
    cuda_free_host((void**)&h_work);
    cuda_free_host((void**)&rwork);
    free(iwork);
    free(w);

    if (info)
    {
        printf("magma_zhegvdx_2stage returned : %i\n", info);
        exit(-1);
    }    

    if (m != nv)
    {
        printf("Not all eigen-values are found.\n");
        exit(-1);
    }
}


