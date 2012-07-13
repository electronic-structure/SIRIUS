#ifndef __LINALG_CPU_H__
#define __LINALG_CPU_H__

#include <stdint.h>
#include "config.h"

extern "C" void FORTRAN(zgemm)(const char *transa, const char *transb, int32_t *m, int32_t *n, 
                               int32_t *k, complex16 *alpha, complex16 *a, int32_t *lda, 
                               complex16 *b, int32_t *ldb, complex16 *beta, complex16 *c, 
                               int32_t *ldc, int32_t transalen, int32_t transblen);

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


extern "C" void FORTRAN(dptsv)(int32_t *n, int32_t *nrhs, double *d, double *e, double *b, int32_t *ldb, 
                               int32_t *info);

extern "C" void FORTRAN(dgtsv)(int32_t *n, int32_t *nrhs, double *dl, double *d, double *du, double *b, 
                               int32_t *ldb, int32_t *info);

int dgtsv(int n, int nrhs, double *dl, double *d, double *du, double *b, int ldb);

#endif // __LINALG_CPU_H__

