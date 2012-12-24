#ifndef __LINALG_CPU_H__
#define __LINALG_CPU_H__

//#include <stdint.h>
#include "config.h"

/*
    matrix-matrix operations
*/
extern "C" void FORTRAN(zgemm)(const char* transa, const char* transb, int4* m, int4* n, int4* k, 
                               complex16* alpha, complex16* a, int4* lda, complex16* b, int4* ldb, 
                               complex16* beta, complex16* c, int4* ldc, int4 transalen, int4 transblen);

extern "C" void FORTRAN(dgemm)(const char* transa, const char* transb, int4* m, int4* n, int4* k, 
                               real8* alpha, real8* a, int4* lda, real8* b, int4* ldb, 
                               real8* beta,real8* c, int4* ldc, int4 transalen, int4 transblen);

extern "C" void FORTRAN(zhemm)(const char *side, const char* uplo, int4* m, int4* n, 
                               complex16* alpha, complex16* a, int4* lda, complex16* b,
                               int4* ldb, complex16* beta, complex16* c, int4* ldc,
                               int4 sidelen, int4 uplolen);

/*
    eigen-value problem
*/
extern "C" void FORTRAN(zhegvx)(int4* itype, const char* jobz, const char* range, const char* uplo, 
                                int4* n, complex16* a, int4* lda, complex16* b, int4* ldb, real8* vl, 
                                real8* vu, int4* il, int4* iu, real8* abstol, int4* m, real8* w, complex16* z,
                                int4* ldz, complex16* work, int4* lwork, real8* rwork, int4* iwork, int4* ifail, 
                                int4* info, int4 jobzlen, int4 rangelen, int4 uplolen);

extern "C" int4 FORTRAN(ilaenv)(int4* ispec, const char* name, const char* opts, int4* n1, int4* n2, int4* n3, 
                                int4* n4, int4 namelen, int4 optslen);

extern "C" void FORTRAN(zheev)(const char* jobz, const char* uplo, int4* n, complex16* a,
                               int4* lda, real8* w, complex16* work, int4* lwork, real8* rwork,
                               int4* info, int4 jobzlen, int4 uplolen);

extern "C" void FORTRAN(zheevd)(const char* jobz, const char* uplo, int4* n, complex16* a,
                                int4* lda, real8* w, complex16* work, int4* lwork, real8* rwork,
                                int4* lrwork, int4* iwork, int4* liwork, int4* info, int4 jobzlen, int4 uplolen);




extern "C" void FORTRAN(dptsv)(int32_t *n, int32_t *nrhs, real8* d, real8* *e, real8* b, int32_t *ldb, int32_t *info);

extern "C" void FORTRAN(dgtsv)(int32_t *n, int32_t *nrhs, double *dl, double *d, double *du, double *b, 
                               int32_t *ldb, int32_t *info);

extern "C" void FORTRAN(zgtsv)(int32_t *n, int32_t *nrhs, complex16* dl, complex16* d, complex16* du, complex16* b, 
                               int32_t *ldb, int32_t *info);

extern "C" void FORTRAN(dgesv)(int4* n, int4* nrhs, real8* a, int4* lda, int4* ipiv, real8* b, int4* ldb, int4* info);

extern "C" void FORTRAN(zgesv)(int4* n, int4* nrhs, complex16* a, int4* lda, int4* ipiv, complex16* b, int4* ldb, int4* info);


/* 
    BLACS and ScaLAPACK related functions
*/
extern "C" int Csys2blacs_handle(MPI_Comm SysCtxt);
extern "C" MPI_Comm Cblacs2sys_handle(int BlacsCtxt);
extern "C" void Cblacs_gridinit(int* ConTxt, const char* order, int nprow, int npcol);
extern "C" void Cblacs_gridmap(int* ConTxt, int* usermap, int ldup, int nprow0, int npcol0);
extern "C" void Cblacs_gridinfo(int ConTxt, int* nprow, int* npcol, int* myrow, int* mycol);
extern "C" void Cfree_blacs_system_handle(int ISysCtxt);
extern "C" void Cblacs_barrier(int ConTxt, const char* scope);

extern "C" void FORTRAN(descinit)(int4* desc, int4* m, int4* n, int4* mb, int4* nb, int4* irsrc, int4* icsrc, 
                                  int4* ictxt, int4* lld, int4* info);

extern "C" void FORTRAN(pzhegvx)(int4* ibtype, const char* jobz, const char* range, const char* uplo, int4* n, 
                        complex16* a, int4* ia, int4* ja, int4* desca, complex16* b, int4* ib, int4* jb, int4* descb, 
                        real8* vl, real8* vu, int4* il, int4* iu, real8* abstol, int4* m, int4* nz, real8* w, 
                        real8* orfac, complex16* z, int4* iz, int4* jz, int4* descz, complex16* work, int4* lwork, 
                        real8* rwork, int4* lrwork, int4* iwork, int4* liwork, int4* ifail, int4* iclustr, real8* gap,
                        int4* info, int4 jobz_len, int4 range_len, int4 uplo_len);

extern "C" void FORTRAN(pzheevd)(const char* jobz, const char* uplo, int4* n, complex16* a, int4* ia, int4* ja, 
                                 int4* desca, real8* w, complex16* z, int4* iz, int4* jz, int4* descz, 
                                 complex16* work, int4* lwork, real8* rwork, int4* lrwork, int4* iwork, int4* liwork, 
                                 int4* info, int4 jobz_len, int4 uplo_len);

extern "C" int4 FORTRAN(numroc)(int4* n, int4* nb, int4* iproc, int4* isrcproc, int4* nprocs);

extern "C" int4 FORTRAN(pjlaenv)(int4* ictxt, int4* ispec, const char* name, const char* opts, int4* n1, int4* n2, 
                                 int4* n3, int4* n4, int4 namelen, int4 optslen);

extern "C" int4 FORTRAN(iceil)(int4* inum, int4* idenom);

#endif // __LINALG_CPU_H__

