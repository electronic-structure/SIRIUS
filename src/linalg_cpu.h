#ifndef __LINALG_CPU_H__
#define __LINALG_CPU_H__

//#include <stdint.h>
#include "config.h"

/*
    matrix-matrix operations
*/
extern "C" void FORTRAN(zgemm)(const char* transa, const char* transb, int32_t* m, int32_t* n, int32_t* k, 
                               complex16* alpha, complex16* a, int32_t* lda, complex16* b, int32_t* ldb, 
                               complex16* beta, complex16* c, int32_t* ldc, int32_t transalen, int32_t transblen);

extern "C" void FORTRAN(dgemm)(const char* transa, const char* transb, int32_t* m, int32_t* n, int32_t* k, 
                               real8* alpha, real8* a, int32_t* lda, real8* b, int32_t* ldb, 
                               real8* beta,real8* c, int32_t* ldc, int32_t transalen, int32_t transblen);

extern "C" void FORTRAN(zhemm)(const char *side, const char* uplo, int32_t* m, int32_t* n, 
                               complex16* alpha, complex16* a, int32_t* lda, complex16* b,
                               int32_t* ldb, complex16* beta, complex16* c, int32_t* ldc,
                               int32_t sidelen, int32_t uplolen);

/*
    eigen-value problem
*/
extern "C" void FORTRAN(zhegvx)(int32_t* itype, const char* jobz, const char* range, const char* uplo, 
                                int32_t* n, complex16* a, int32_t* lda, complex16* b, int32_t* ldb, real8* vl, 
                                real8* vu, int32_t* il, int32_t* iu, real8* abstol, int32_t* m, real8* w, complex16* z,
                                int32_t* ldz, complex16* work, int32_t* lwork, real8* rwork, int32_t* iwork, int32_t* ifail, 
                                int32_t* info, int32_t jobzlen, int32_t rangelen, int32_t uplolen);

extern "C" int32_t FORTRAN(ilaenv)(int32_t* ispec, const char* name, const char* opts, int32_t* n1, int32_t* n2, int32_t* n3, 
                                int32_t* n4, int32_t namelen, int32_t optslen);

extern "C" void FORTRAN(zheev)(const char* jobz, const char* uplo, int32_t* n, complex16* a,
                               int32_t* lda, real8* w, complex16* work, int32_t* lwork, real8* rwork,
                               int32_t* info, int32_t jobzlen, int32_t uplolen);

extern "C" void FORTRAN(zheevd)(const char* jobz, const char* uplo, int32_t* n, complex16* a,
                                int32_t* lda, real8* w, complex16* work, int32_t* lwork, real8* rwork,
                                int32_t* lrwork, int32_t* iwork, int32_t* liwork, int32_t* info, int32_t jobzlen, int32_t uplolen);




extern "C" void FORTRAN(dptsv)(int32_t *n, int32_t *nrhs, real8* d, real8* *e, real8* b, int32_t *ldb, int32_t *info);

extern "C" void FORTRAN(dgtsv)(int32_t *n, int32_t *nrhs, double *dl, double *d, double *du, double *b, 
                               int32_t *ldb, int32_t *info);

extern "C" void FORTRAN(zgtsv)(int32_t *n, int32_t *nrhs, complex16* dl, complex16* d, complex16* du, complex16* b, 
                               int32_t *ldb, int32_t *info);

extern "C" void FORTRAN(dgesv)(int32_t* n, int32_t* nrhs, real8* a, int32_t* lda, int32_t* ipiv, real8* b, int32_t* ldb, int32_t* info);

extern "C" void FORTRAN(zgesv)(int32_t* n, int32_t* nrhs, complex16* a, int32_t* lda, int32_t* ipiv, complex16* b, int32_t* ldb, int32_t* info);

extern "C" void FORTRAN(dgetrf)(int32_t* m, int32_t* n, real8* a, int32_t* lda, int32_t* ipiv, int32_t* info);

extern "C" void FORTRAN(zgetrf)(int32_t* m, int32_t* n, complex16* a, int32_t* lda, int32_t* ipiv, int32_t* info);

extern "C" void FORTRAN(dgetri)(int32_t* n, real8* a, int32_t* lda, int32_t* ipiv, real8* work, int32_t* lwork, int32_t* info);

extern "C" void FORTRAN(zgetri)(int32_t* n, complex16* a, int32_t* lda, int32_t* ipiv, complex16* work, int32_t* lwork, int32_t* info);




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

extern "C" void FORTRAN(descinit)(int32_t* desc, int32_t* m, int32_t* n, int32_t* mb, int32_t* nb, int32_t* irsrc, int32_t* icsrc, 
                                  int32_t* ictxt, int32_t* lld, int32_t* info);

extern "C" void FORTRAN(pztranc)(int32_t* m, int32_t* n, complex16* alpha, complex16* a, int32_t* ia, int32_t* ja, int32_t* desca,
                                 complex16* beta, complex16* c, int32_t* ic, int32_t* jc,int32_t* descc);

extern "C" void FORTRAN(pzhegvx)(int32_t* ibtype, const char* jobz, const char* range, const char* uplo, int32_t* n, 
                                 complex16* a, int32_t* ia, int32_t* ja, int32_t* desca, 
                                 complex16* b, int32_t* ib, int32_t* jb, int32_t* descb, 
                                 real8* vl, real8* vu, 
                                 int32_t* il, int32_t* iu, 
                                 real8* abstol, 
                                 int32_t* m, int32_t* nz, real8* w, real8* orfac, 
                                 complex16* z, int32_t* iz, int32_t* jz, int32_t* descz, 
                                 complex16* work, int32_t* lwork, 
                                 real8* rwork, int32_t* lrwork, 
                                 int32_t* iwork, int32_t* liwork, 
                                 int32_t* ifail, int32_t* iclustr, real8* gap, int32_t* info, 
                                 int32_t jobz_len, int32_t range_len, int32_t uplo_len);

extern "C" void FORTRAN(pzheevd)(const char* jobz, const char* uplo, int32_t* n, 
                                 complex16* a, int32_t* ia, int32_t* ja, int32_t* desca, 
                                 real8* w, 
                                 complex16* z, int32_t* iz, int32_t* jz, int32_t* descz, 
                                 complex16* work, int32_t* lwork, real8* rwork, int32_t* lrwork, int32_t* iwork, 
                                 int32_t* liwork, int32_t* info, int32_t jobz_len, int32_t uplo_len);

extern "C" void FORTRAN(pzgemm)(const char* transa, const char* transb, 
                                int32_t* m, int32_t* n, int32_t* k, 
                                complex16* aplha,
                                complex16* a, int32_t* ia, int32_t* ja, int32_t* desca, 
                                complex16* b, int32_t* ib, int32_t* jb, int32_t* descb,
                                complex16* beta,
                                complex16* c, int32_t* ic, int32_t* jc, int32_t* descc,
                                int32_t transa_len, int32_t transb_len);

extern "C" int32_t FORTRAN(numroc)(int32_t* n, int32_t* nb, int32_t* iproc, int32_t* isrcproc, int32_t* nprocs);

extern "C" int32_t FORTRAN(indxl2g)(int32_t* indxloc, int32_t* nb, int32_t* iproc, int32_t* isrcproc, int32_t* nprocs);

extern "C" int32_t FORTRAN(pjlaenv)(int32_t* ictxt, int32_t* ispec, const char* name, const char* opts, int32_t* n1, int32_t* n2, 
                                 int32_t* n3, int32_t* n4, int32_t namelen, int32_t optslen);

extern "C" int32_t FORTRAN(iceil)(int32_t* inum, int32_t* idenom);

extern "C" void FORTRAN(elpa_cholesky_complex)(int32_t* na, complex16* a, int32_t* lda, int32_t* nblk, int32_t* mpi_comm_rows, 
                                               int32_t* mpi_comm_cols);

extern "C" void FORTRAN(elpa_invert_trm_complex)(int32_t* na, complex16* a, int32_t* lda, int32_t* nblk, int32_t* mpi_comm_rows, 
                                                 int32_t* mpi_comm_cols);

extern "C" void FORTRAN(elpa_mult_ah_b_complex)(const char* uplo_a, const char* uplo_c, int32_t* na, int32_t* ncb, 
                                                complex16* a, int32_t* lda, complex16* b, int32_t* ldb, int32_t* nblk, 
                                                int32_t* mpi_comm_rows, int32_t* mpi_comm_cols, complex16* c, int32_t* ldc,
                                                int32_t uplo_a_len, int32_t uplo_c_len);

extern "C" void FORTRAN(elpa_solve_evp_complex)(int32_t* na, int32_t* nev, complex16* a, int32_t* lda, real8* ev, complex16* q, 
                                                int32_t* ldq, int32_t* nblk, int32_t* mpi_comm_rows, int32_t* mpi_comm_cols);

extern "C" void FORTRAN(elpa_solve_evp_complex_2stage)(int32_t* na, int32_t* nev, complex16* a, int32_t* lda, real8* ev, 
                                                       complex16* q, int32_t* ldq, int32_t* nblk, int32_t* mpi_comm_rows, 
                                                       int32_t* mpi_comm_cols, int32_t* mpi_comm_all);

#endif // __LINALG_CPU_H__

