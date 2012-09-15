#ifndef __LINALG_CPU_H__
#define __LINALG_CPU_H__

#include <stdint.h>
#include "config.h"

extern "C" void FORTRAN(zgemm)(const char* transa, const char* transb, int4* m, int4* n, int4* k, 
                               complex16* alpha, complex16* a, int4* lda, complex16* b, int4* ldb, 
                               complex16* beta, complex16* c, int4* ldc, int4 transalen, int4 transblen);

extern "C" void FORTRAN(zhemm)(const char *side, const char *uplo, int32_t *m, int32_t *n, 
                               complex16 *alpha, complex16 *a, int32_t *lda, complex16 *b,
                               int32_t *ldb, complex16 *beta, complex16 *c, int32_t *ldc,
                               int32_t sidelen, int32_t uplolen);
    
extern "C" void FORTRAN(zhegvx)(int32_t *itype, const char *jobz, const char *range, 
                                const char *uplo, int32_t *n, complex16 *a, int32_t *lda,
                                complex16 *b, int32_t *ldb, double *vl, double *vu, int32_t *il,
                                int32_t *iu, double *abstol, int32_t *m, double *w, complex16 *z,
                                int32_t *ldz, complex16 *work, int32_t *lwork, double *rwork,
                                int32_t *iwork, int32_t *ifail, int32_t *info, int32_t jobzlen,
                                int32_t rangelen, int32_t uplolen);

extern "C" int32_t FORTRAN(ilaenv)(int32_t *ispec, const char *name, const char *opts, int32_t *n1,
                                   int32_t *n2, int32_t *n3, int32_t *n4, int32_t namelen, 
                                   int32_t optslen);

extern "C" void FORTRAN(zheev)(const char *jobz, const char *uplo, int32_t *n, complex16 *a,
                               int32_t *lda, double *w, double *work, int32_t *lwork, double *rwork,
                               int32_t *info, int32_t jobzlen, int32_t uplolen);

extern "C" void FORTRAN(zcopy)(int32_t *n, complex16 *zx, int32_t *incx, complex16 *zy, int32_t *incy);

void zcopy(int32_t n, complex16 *zx, int32_t incx, complex16 *zy, int32_t incy);


extern "C" void FORTRAN(zgemv)(const char *transa, int32_t *m, int32_t *n, complex16 *alpha, complex16 *a,
                               int32_t *lda, complex16 *x, int32_t *incx, complex16 *beta, complex16 *y,
                               int32_t *incy, int32_t translen);


extern "C" void FORTRAN(dptsv)(int32_t *n, int32_t *nrhs, real8* d, real8* *e, real8* b, int32_t *ldb, int32_t *info);

extern "C" void FORTRAN(dgtsv)(int32_t *n, int32_t *nrhs, double *dl, double *d, double *du, double *b, 
                               int32_t *ldb, int32_t *info);

extern "C" void FORTRAN(zgtsv)(int32_t *n, int32_t *nrhs, complex16* dl, complex16* d, complex16* du, complex16* b, 
                               int32_t *ldb, int32_t *info);

template<typename T> int gtsv(int n, int nrhs, T* dl, T* d, T* du, T* b, int ldb);

template<> int gtsv<double>(int n, int nrhs, double *dl, double *d, double *du, double *b, int ldb)
{
    int info;

    FORTRAN(dgtsv)(&n, &nrhs, dl, d, du, b, &ldb, &info);
            
    return info;
}

template<> int gtsv<complex16>(int n, int nrhs, complex16* dl, complex16* d, complex16* du, complex16* b, int ldb)
{
    int4 info;                   

    FORTRAN(zgtsv)(&n, &nrhs, dl, d, du, b, &ldb, &info);
            
    return info;                
}

extern "C" void FORTRAN(dgesv)(int4* n, int4* nrhs, real8* a, int4* lda, int4* ipiv, real8* b, int4* ldb, int4* info);

extern "C" void FORTRAN(zgesv)(int4* n, int4* nrhs, complex16* a, int4* lda, int4* ipiv, complex16* b, int4* ldb, int4* info);

template <typename T> int gesv(int4 n, int4 nrhs, T* a, int4 lda, T* b, int4 ldb);

template<> int gesv<real8>(int4 n, int4 nrhs, real8* a, int4 lda, real8* b, int4 ldb)
{
    int4 info;
    std::vector<int4> ipiv(n);

    FORTRAN(dgesv)(&n, &nrhs, a, &lda, &ipiv[0], b, &ldb, &info);

    return info;
}

template<> int gesv<complex16>(int4 n, int4 nrhs, complex16* a, int4 lda, complex16* b, int4 ldb)
{
    int4 info;
    std::vector<int4> ipiv(n);

    FORTRAN(zgesv)(&n, &nrhs, a, &lda, &ipiv[0], b, &ldb, &info);

    return info;
}

extern "C" void FORTRAN(dgemm)(const char* transa, const char* transb, int4* m, int4* n, int4* k, real8* alpha, real8* a,
                               int4* lda, real8* b, int4* ldb, real8* beta, real8* c, int4* ldc, int4 transalen, int4 transblen);

#endif // __LINALG_CPU_H__

