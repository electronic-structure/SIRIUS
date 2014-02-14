#include <plasma.h>
#include <mkl_service.h>
//#include <plasma_threadsetting.h>

extern "C" void plasma_init(int num_cores)
{
    PLASMA_Init(num_cores);
}

extern "C" void plasma_zheevd_wrapper(int32_t matrix_size, void* a, int32_t lda, void* z,
                                      int32_t ldz, double* eval)
{
    PLASMA_desc* descT;

    PLASMA_Alloc_Workspace_zheevd(matrix_size, matrix_size, &descT);

    int info = PLASMA_zheevd(PlasmaVec, PlasmaUpper, matrix_size, (PLASMA_Complex64_t*)a, lda, eval, descT, (PLASMA_Complex64_t*)z, ldz);
    if (info != 0)
    {
        printf("erorr calling PLASMA_zheevd\n");
        exit(0);
    }

    PLASMA_Dealloc_Handle_Tile(&descT);
}


extern "C" void plasma_set_num_threads(int num_threads)
{
    //plasma_setlapack_numthreads(num_threads);
    mkl_set_num_threads(num_threads);
}

