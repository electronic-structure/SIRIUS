// This file is part of SIRIUS
//
// Copyright (c) 2013 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef __LINALG_H__
#define __LINALG_H__

/** \file linalg.h
  *
  * \brief Contains linear algebra bindings.
  *
  */

#include <stdint.h>
#include "config.h"
#include "error_handling.h"

/*
 *  matrix-vector operations
 */
extern "C" void FORTRAN(zgemv)(const char* trans, int32_t* m, int32_t* n, double_complex* alpha, 
                               double_complex* a, int32_t* lda, double_complex* x, int32_t* incx,
                               double_complex* beta, double_complex* y, int32_t* incy, int32_t translen);

/*
 *  matrix-matrix operations
 */
extern "C" void FORTRAN(zgemm)(const char* transa, const char* transb, int32_t* m, int32_t* n, int32_t* k, 
                               double_complex* alpha, double_complex* a, int32_t* lda, double_complex* b, int32_t* ldb, 
                               double_complex* beta, double_complex* c, int32_t* ldc, int32_t transalen, int32_t transblen);

extern "C" void FORTRAN(dgemm)(const char* transa, const char* transb, int32_t* m, int32_t* n, int32_t* k, 
                               double* alpha, double* a, int32_t* lda, double* b, int32_t* ldb, 
                               double* beta,double* c, int32_t* ldc, int32_t transalen, int32_t transblen);

extern "C" void FORTRAN(zhemm)(const char *side, const char* uplo, int32_t* m, int32_t* n, 
                               double_complex* alpha, double_complex* a, int32_t* lda, double_complex* b,
                               int32_t* ldb, double_complex* beta, double_complex* c, int32_t* ldc,
                               int32_t sidelen, int32_t uplolen);

/*
 *  eigen-value problem
 */
extern "C" void FORTRAN(zhegvx)(int32_t* itype, const char* jobz, const char* range, const char* uplo, 
                                int32_t* n, double_complex* a, int32_t* lda, double_complex* b, int32_t* ldb, double* vl, 
                                double* vu, int32_t* il, int32_t* iu, double* abstol, int32_t* m, double* w, double_complex* z,
                                int32_t* ldz, double_complex* work, int32_t* lwork, double* rwork, int32_t* iwork, int32_t* ifail, 
                                int32_t* info, int32_t jobzlen, int32_t rangelen, int32_t uplolen);

extern "C" int32_t FORTRAN(ilaenv)(int32_t* ispec, const char* name, const char* opts, int32_t* n1, int32_t* n2, int32_t* n3, 
                                int32_t* n4, int32_t namelen, int32_t optslen);

extern "C" void FORTRAN(zheev)(const char* jobz, const char* uplo, int32_t* n, double_complex* a,
                               int32_t* lda, double* w, double_complex* work, int32_t* lwork, double* rwork,
                               int32_t* info, int32_t jobzlen, int32_t uplolen);

extern "C" void FORTRAN(zheevd)(const char* jobz, const char* uplo, int32_t* n, double_complex* a,
                                int32_t* lda, double* w, double_complex* work, int32_t* lwork, double* rwork,
                                int32_t* lrwork, int32_t* iwork, int32_t* liwork, int32_t* info, int32_t jobzlen, int32_t uplolen);




extern "C" void FORTRAN(dptsv)(int32_t *n, int32_t *nrhs, double* d, double* *e, double* b, int32_t *ldb, int32_t *info);

extern "C" void FORTRAN(dgtsv)(int32_t *n, int32_t *nrhs, double *dl, double *d, double *du, double *b, 
                               int32_t *ldb, int32_t *info);

extern "C" void FORTRAN(zgtsv)(int32_t *n, int32_t *nrhs, double_complex* dl, double_complex* d, double_complex* du, double_complex* b, 
                               int32_t *ldb, int32_t *info);

extern "C" void FORTRAN(dgesv)(int32_t* n, int32_t* nrhs, double* a, int32_t* lda, int32_t* ipiv, double* b, int32_t* ldb, int32_t* info);

extern "C" void FORTRAN(zgesv)(int32_t* n, int32_t* nrhs, double_complex* a, int32_t* lda, int32_t* ipiv, double_complex* b, int32_t* ldb, int32_t* info);

extern "C" void FORTRAN(dgetrf)(int32_t* m, int32_t* n, double* a, int32_t* lda, int32_t* ipiv, int32_t* info);

extern "C" void FORTRAN(zgetrf)(int32_t* m, int32_t* n, double_complex* a, int32_t* lda, int32_t* ipiv, int32_t* info);

extern "C" void FORTRAN(dgetri)(int32_t* n, double* a, int32_t* lda, int32_t* ipiv, double* work, int32_t* lwork, int32_t* info);

extern "C" void FORTRAN(zgetri)(int32_t* n, double_complex* a, int32_t* lda, int32_t* ipiv, double_complex* work, int32_t* lwork, int32_t* info);


/* 
 *  BLACS and ScaLAPACK related functions
 */
#ifdef _SCALAPACK_
extern "C" int Csys2blacs_handle(MPI_Comm SysCtxt);
extern "C" MPI_Comm Cblacs2sys_handle(int BlacsCtxt);
extern "C" void Cblacs_gridinit(int* ConTxt, const char* order, int nprow, int npcol);
extern "C" void Cblacs_gridmap(int* ConTxt, int* usermap, int ldup, int nprow0, int npcol0);
extern "C" void Cblacs_gridinfo(int ConTxt, int* nprow, int* npcol, int* myrow, int* mycol);
extern "C" void Cfree_blacs_system_handle(int ISysCtxt);
extern "C" void Cblacs_barrier(int ConTxt, const char* scope);

extern "C" void FORTRAN(descinit)(int32_t* desc, int32_t* m, int32_t* n, int32_t* mb, int32_t* nb, int32_t* irsrc, int32_t* icsrc, 
                                  int32_t* ictxt, int32_t* lld, int32_t* info);

extern "C" void FORTRAN(pztranc)(int32_t* m, int32_t* n, double_complex* alpha, double_complex* a, int32_t* ia, int32_t* ja, int32_t* desca,
                                 double_complex* beta, double_complex* c, int32_t* ic, int32_t* jc,int32_t* descc);

extern "C" void FORTRAN(pzhegvx)(int32_t* ibtype, const char* jobz, const char* range, const char* uplo, int32_t* n, 
                                 double_complex* a, int32_t* ia, int32_t* ja, int32_t* desca, 
                                 double_complex* b, int32_t* ib, int32_t* jb, int32_t* descb, 
                                 double* vl, double* vu, 
                                 int32_t* il, int32_t* iu, 
                                 double* abstol, 
                                 int32_t* m, int32_t* nz, double* w, double* orfac, 
                                 double_complex* z, int32_t* iz, int32_t* jz, int32_t* descz, 
                                 double_complex* work, int32_t* lwork, 
                                 double* rwork, int32_t* lrwork, 
                                 int32_t* iwork, int32_t* liwork, 
                                 int32_t* ifail, int32_t* iclustr, double* gap, int32_t* info, 
                                 int32_t jobz_len, int32_t range_len, int32_t uplo_len);

extern "C" void FORTRAN(pzheevd)(const char* jobz, const char* uplo, int32_t* n, 
                                 double_complex* a, int32_t* ia, int32_t* ja, int32_t* desca, 
                                 double* w, 
                                 double_complex* z, int32_t* iz, int32_t* jz, int32_t* descz, 
                                 double_complex* work, int32_t* lwork, double* rwork, int32_t* lrwork, int32_t* iwork, 
                                 int32_t* liwork, int32_t* info, int32_t jobz_len, int32_t uplo_len);

extern "C" void FORTRAN(pzgemm)(const char* transa, const char* transb, 
                                int32_t* m, int32_t* n, int32_t* k, 
                                double_complex* aplha,
                                double_complex* a, int32_t* ia, int32_t* ja, int32_t* desca, 
                                double_complex* b, int32_t* ib, int32_t* jb, int32_t* descb,
                                double_complex* beta,
                                double_complex* c, int32_t* ic, int32_t* jc, int32_t* descc,
                                int32_t transa_len, int32_t transb_len);

extern "C" int32_t FORTRAN(numroc)(int32_t* n, int32_t* nb, int32_t* iproc, int32_t* isrcproc, int32_t* nprocs);

extern "C" int32_t FORTRAN(indxl2g)(int32_t* indxloc, int32_t* nb, int32_t* iproc, int32_t* isrcproc, int32_t* nprocs);

extern "C" int32_t FORTRAN(pjlaenv)(int32_t* ictxt, int32_t* ispec, const char* name, const char* opts, int32_t* n1, int32_t* n2, 
                                 int32_t* n3, int32_t* n4, int32_t namelen, int32_t optslen);

extern "C" int32_t FORTRAN(iceil)(int32_t* inum, int32_t* idenom);
#endif

#ifdef _ELPA_
extern "C" void FORTRAN(elpa_cholesky_complex)(int32_t* na, double_complex* a, int32_t* lda, int32_t* nblk, int32_t* mpi_comm_rows, 
                                               int32_t* mpi_comm_cols);

extern "C" void FORTRAN(elpa_invert_trm_complex)(int32_t* na, double_complex* a, int32_t* lda, int32_t* nblk, int32_t* mpi_comm_rows, 
                                                 int32_t* mpi_comm_cols);

extern "C" void FORTRAN(elpa_mult_ah_b_complex)(const char* uplo_a, const char* uplo_c, int32_t* na, int32_t* ncb, 
                                                double_complex* a, int32_t* lda, double_complex* b, int32_t* ldb, int32_t* nblk, 
                                                int32_t* mpi_comm_rows, int32_t* mpi_comm_cols, double_complex* c, int32_t* ldc,
                                                int32_t uplo_a_len, int32_t uplo_c_len);

extern "C" void FORTRAN(elpa_solve_evp_complex)(int32_t* na, int32_t* nev, double_complex* a, int32_t* lda, double* ev, 
                                                double_complex* q, int32_t* ldq, int32_t* nblk, int32_t* mpi_comm_rows, 
                                                int32_t* mpi_comm_cols);

extern "C" void FORTRAN(elpa_solve_evp_complex_2stage)(int32_t* na, int32_t* nev, double_complex* a, int32_t* lda, double* ev, 
                                                       double_complex* q, int32_t* ldq, int32_t* nblk, int32_t* mpi_comm_rows, 
                                                       int32_t* mpi_comm_cols, int32_t* mpi_comm_all);
#endif

#include "lapack.h"
#include "dmatrix.h"
#include "blas.h"
#include "evp_solver.h"

#endif // __LINALG_H__

