#ifndef __LINALG_GPU_H__
#define __LINALG_GPU_H__

//* extern "C" void magma_zhegvdx_2stage_wrapper(int32_t matrix_size, int32_t nv, void* a, int32_t lda, 
//*                                              void* b, int32_t ldb, double* eval, void* z, int32_t ldz);
//* 
//* extern "C" void gpu_zgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
//*                           complex16 alpha, complex16 *a, int32_t lda, complex16 *b, 
//*                           int32_t ldb, complex16 beta, complex16 *c, int32_t ldc);
//* 
//* extern "C" void gpu_zhegvx(int32_t n, int32_t nv, double abstol, void *a, void *b,
//*                            double *eval, void *z, int32_t ldz);
//*    
#endif // __LINALG_GPU_H__

