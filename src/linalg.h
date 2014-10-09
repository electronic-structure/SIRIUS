// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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

/** \file linalg.h
 *
 *  \brief Linear algebra interface.
 */

#ifndef __LINALG_H__
#define __LINALG_H__

#include <stdint.h>
#include <complex>
#include <mpi.h>
#include "typedefs.h"
//#include "config.h"
#include "error_handling.h"

#define FORTRAN(x) x##_

// Assume a 32-bit integer BLAS/LAPACK etc.
typedef int32_t ftn_int;

// Assume a 32-bit integer for implicit string length arguments
typedef int32_t ftn_len;

typedef double ftn_double;

typedef std::complex<double> ftn_double_complex;

typedef char const* ftn_char;

extern "C" {

/*
 *  matrix-vector operations
 */

void FORTRAN(zgemv)(ftn_char trans, ftn_int* m, ftn_int* n, ftn_double_complex* alpha, 
                    ftn_double_complex* A, ftn_int* lda, ftn_double_complex* X, ftn_int* incx,
                    ftn_double_complex* beta, ftn_double_complex* Y, ftn_int* incy, ftn_len trans_len);

/*
 *  matrix-matrix operations
 */

void FORTRAN(dgemm)(ftn_char transa, ftn_char transb, ftn_int* m, ftn_int* n, ftn_int* k, 
                    ftn_double* alpha, ftn_double* A, ftn_int* lda, ftn_double* B, ftn_int* ldb, 
                    ftn_double* beta, ftn_double* C, ftn_int* ldc, ftn_len transa_len, ftn_len transb_len);

void FORTRAN(zgemm)(ftn_char transa, ftn_char transb, ftn_int* m, ftn_int* n, ftn_int* k, 
                    ftn_double_complex* alpha, ftn_double_complex* A, ftn_int* lda, ftn_double_complex* B, ftn_int* ldb, 
                    ftn_double_complex* beta, ftn_double_complex* C, ftn_int* ldc, ftn_len transa_len, ftn_len transb_len);

void FORTRAN(dgetrf)(ftn_int* m, ftn_int* n, ftn_double* A, ftn_int* lda, ftn_int* ipiv, ftn_int* info);

void FORTRAN(dgetri)(ftn_int* n, ftn_double* A, ftn_int* lda, ftn_int* ipiv, ftn_double* work, ftn_int* lwork, ftn_int* info);

void FORTRAN(zgetrf)(ftn_int* m, ftn_int* n, ftn_double_complex* A, ftn_int* lda, ftn_int* ipiv, ftn_int* info);

void FORTRAN(zgetri)(ftn_int* n, ftn_double_complex* A, ftn_int* lda, ftn_int* ipiv, ftn_double_complex* work, ftn_int* lwork,
                     ftn_int* info);
#ifdef _SCALAPACK_
void FORTRAN(pzgemm)(ftn_char transa, ftn_char transb, ftn_int* m, ftn_int* n, ftn_int* k, ftn_double_complex* aplha,
                     ftn_double_complex* A, ftn_int* ia, ftn_int* ja, ftn_int* desca, 
                     ftn_double_complex* B, ftn_int* ib, ftn_int* jb, ftn_int* descb, ftn_double_complex* beta,
                     ftn_double_complex* C, ftn_int* ic, ftn_int* jc, ftn_int* descc,
                     ftn_len transa_len, ftn_len transb_len);

void FORTRAN(pzgetrf)(ftn_int* m, ftn_int* n, ftn_double_complex* A, ftn_int* ia, ftn_int* ja, ftn_int* desca, ftn_int* ipiv,
                      ftn_int* info);

void FORTRAN(pzgetri)(ftn_int* n, ftn_double_complex* A, ftn_int* ia, ftn_int* ja, ftn_int* desca, ftn_int* ipiv,
                      ftn_double_complex* work, ftn_int* lwork, ftn_int* iwork, ftn_int* liwork, ftn_int* info);
#endif

}




extern "C" void FORTRAN(zhemm)(const char *side, const char* uplo, ftn_int* m, ftn_int* n, 
                               ftn_double_complex* alpha, ftn_double_complex* a, ftn_int* lda, ftn_double_complex* b,
                               ftn_int* ldb, ftn_double_complex* beta, ftn_double_complex* c, ftn_int* ldc,
                               int32_t sidelen, int32_t uplolen);

/*
 *  eigen-value problem
 */
extern "C" void FORTRAN(zhegvx)(ftn_int* itype, const char* jobz, const char* range, const char* uplo, 
                                ftn_int* n, ftn_double_complex* a, ftn_int* lda, ftn_double_complex* b, ftn_int* ldb, ftn_double* vl, 
                                ftn_double* vu, ftn_int* il, ftn_int* iu, ftn_double* abstol, ftn_int* m, ftn_double* w, ftn_double_complex* z,
                                ftn_int* ldz, ftn_double_complex* work, ftn_int* lwork, ftn_double* rwork, ftn_int* iwork, ftn_int* ifail, 
                                ftn_int* info, int32_t jobzlen, int32_t rangelen, int32_t uplolen);

extern "C" int32_t FORTRAN(ilaenv)(ftn_int* ispec, const char* name, const char* opts, ftn_int* n1, ftn_int* n2, ftn_int* n3, 
                                ftn_int* n4, int32_t namelen, int32_t optslen);

extern "C" void FORTRAN(zheev)(const char* jobz, const char* uplo, ftn_int* n, ftn_double_complex* a,
                               ftn_int* lda, ftn_double* w, ftn_double_complex* work, ftn_int* lwork, ftn_double* rwork,
                               ftn_int* info, int32_t jobzlen, int32_t uplolen);

extern "C" void FORTRAN(zheevd)(const char* jobz, const char* uplo, ftn_int* n, ftn_double_complex* a,
                                ftn_int* lda, ftn_double* w, ftn_double_complex* work, ftn_int* lwork, ftn_double* rwork,
                                ftn_int* lrwork, ftn_int* iwork, ftn_int* liwork, ftn_int* info, int32_t jobzlen, int32_t uplolen);




extern "C" void FORTRAN(dptsv)(int32_t *n, int32_t *nrhs, ftn_double* d, ftn_double* *e, ftn_double* b, int32_t *ldb, int32_t *info);

extern "C" void FORTRAN(dgtsv)(int32_t *n, int32_t *nrhs, double *dl, double *d, double *du, double *b, 
                               int32_t *ldb, int32_t *info);

extern "C" void FORTRAN(zgtsv)(int32_t *n, int32_t *nrhs, ftn_double_complex* dl, ftn_double_complex* d, ftn_double_complex* du, ftn_double_complex* b, 
                               int32_t *ldb, int32_t *info);

extern "C" void FORTRAN(dgesv)(ftn_int* n, ftn_int* nrhs, ftn_double* a, ftn_int* lda, ftn_int* ipiv, ftn_double* b, ftn_int* ldb, ftn_int* info);

extern "C" void FORTRAN(zgesv)(ftn_int* n, ftn_int* nrhs, ftn_double_complex* a, ftn_int* lda, ftn_int* ipiv, ftn_double_complex* b, ftn_int* ldb, ftn_int* info);



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
extern "C" void Cblacs_gridexit(int ConTxt);

extern "C" void FORTRAN(descinit)(ftn_int* desc, ftn_int* m, ftn_int* n, ftn_int* mb, ftn_int* nb, ftn_int* irsrc, ftn_int* icsrc, 
                                  ftn_int* ictxt, ftn_int* lld, ftn_int* info);

extern "C" void FORTRAN(pztranc)(ftn_int* m, ftn_int* n, ftn_double_complex* alpha, ftn_double_complex* a, ftn_int* ia, ftn_int* ja, ftn_int* desca,
                                 ftn_double_complex* beta, ftn_double_complex* c, ftn_int* ic, ftn_int* jc,ftn_int* descc);

extern "C" void FORTRAN(pztranu)(ftn_int* m, ftn_int* n, ftn_double_complex* alpha, ftn_double_complex* a, ftn_int* ia, ftn_int* ja, ftn_int* desca,
                                 ftn_double_complex* beta, ftn_double_complex* c, ftn_int* ic, ftn_int* jc,ftn_int* descc);

extern "C" void FORTRAN(pzhegvx)(ftn_int* ibtype, const char* jobz, const char* range, const char* uplo, ftn_int* n, 
                                 ftn_double_complex* a, ftn_int* ia, ftn_int* ja, ftn_int* desca, 
                                 ftn_double_complex* b, ftn_int* ib, ftn_int* jb, ftn_int* descb, 
                                 ftn_double* vl, ftn_double* vu, 
                                 ftn_int* il, ftn_int* iu, 
                                 ftn_double* abstol, 
                                 ftn_int* m, ftn_int* nz, ftn_double* w, ftn_double* orfac, 
                                 ftn_double_complex* z, ftn_int* iz, ftn_int* jz, ftn_int* descz, 
                                 ftn_double_complex* work, ftn_int* lwork, 
                                 ftn_double* rwork, ftn_int* lrwork, 
                                 ftn_int* iwork, ftn_int* liwork, 
                                 ftn_int* ifail, ftn_int* iclustr, ftn_double* gap, ftn_int* info, 
                                 int32_t jobz_len, int32_t range_len, int32_t uplo_len);

extern "C" void FORTRAN(pzheevd)(const char* jobz, const char* uplo, ftn_int* n, 
                                 ftn_double_complex* a, ftn_int* ia, ftn_int* ja, ftn_int* desca, 
                                 ftn_double* w, 
                                 ftn_double_complex* z, ftn_int* iz, ftn_int* jz, ftn_int* descz, 
                                 ftn_double_complex* work, ftn_int* lwork, ftn_double* rwork, ftn_int* lrwork, ftn_int* iwork, 
                                 ftn_int* liwork, ftn_int* info, int32_t jobz_len, int32_t uplo_len);


extern "C" void FORTRAN(pdgemm)(const char* transa, const char* transb, 
                                ftn_int* m, ftn_int* n, ftn_int* k, 
                                ftn_double* aplha,
                                ftn_double* a, ftn_int* ia, ftn_int* ja, ftn_int* desca, 
                                ftn_double* b, ftn_int* ib, ftn_int* jb, ftn_int* descb,
                                ftn_double* beta,
                                ftn_double* c, ftn_int* ic, ftn_int* jc, ftn_int* descc,
                                int32_t transa_len, int32_t transb_len);

extern "C" int32_t FORTRAN(numroc)(ftn_int* n, ftn_int* nb, ftn_int* iproc, ftn_int* isrcproc, ftn_int* nprocs);

extern "C" int32_t FORTRAN(indxl2g)(ftn_int* indxloc, ftn_int* nb, ftn_int* iproc, ftn_int* isrcproc, ftn_int* nprocs);

extern "C" int32_t FORTRAN(pjlaenv)(ftn_int* ictxt, ftn_int* ispec, const char* name, const char* opts, ftn_int* n1, ftn_int* n2, 
                                 ftn_int* n3, ftn_int* n4, int32_t namelen, int32_t optslen);

extern "C" int32_t FORTRAN(iceil)(ftn_int* inum, ftn_int* idenom);
#endif

#ifdef _ELPA_
extern "C" void FORTRAN(elpa_cholesky_complex)(ftn_int* na, ftn_double_complex* a, ftn_int* lda, ftn_int* nblk, ftn_int* mpi_comm_rows, 
                                               ftn_int* mpi_comm_cols);

extern "C" void FORTRAN(elpa_invert_trm_complex)(ftn_int* na, ftn_double_complex* a, ftn_int* lda, ftn_int* nblk, ftn_int* mpi_comm_rows, 
                                                 ftn_int* mpi_comm_cols);

extern "C" void FORTRAN(elpa_mult_ah_b_complex)(const char* uplo_a, const char* uplo_c, ftn_int* na, ftn_int* ncb, 
                                                ftn_double_complex* a, ftn_int* lda, ftn_double_complex* b, ftn_int* ldb, ftn_int* nblk, 
                                                ftn_int* mpi_comm_rows, ftn_int* mpi_comm_cols, ftn_double_complex* c, ftn_int* ldc,
                                                int32_t uplo_a_len, int32_t uplo_c_len);

extern "C" void FORTRAN(elpa_solve_evp_complex)(ftn_int* na, ftn_int* nev, ftn_double_complex* a, ftn_int* lda, ftn_double* ev, 
                                                ftn_double_complex* q, ftn_int* ldq, ftn_int* nblk, ftn_int* mpi_comm_rows, 
                                                ftn_int* mpi_comm_cols);

extern "C" void FORTRAN(elpa_solve_evp_complex_2stage)(ftn_int* na, ftn_int* nev, ftn_double_complex* a, ftn_int* lda, ftn_double* ev, 
                                                       ftn_double_complex* q, ftn_int* ldq, ftn_int* nblk, ftn_int* mpi_comm_rows, 
                                                       ftn_int* mpi_comm_cols, ftn_int* mpi_comm_all);
#endif

#include "lapack.h"
#include "dmatrix.h"
#include "blas.h"
#include "evp_solver.h"

class linalg_base
{
    public:
        
        static ftn_int ilaenv(ftn_int ispec, std::string const& name, std::string const& opts, ftn_int n1, ftn_int n2, 
                              ftn_int n3, ftn_int n4)
        {
            return FORTRAN(ilaenv)(&ispec, name.c_str(), opts.c_str(), &n1, &n2, &n3, &n4, (ftn_len)name.length(), 
                                   (ftn_len)opts.length());
        }
        
        #ifdef _SCALAPACK_
        static ftn_int numroc(ftn_int n, ftn_int nb, ftn_int iproc, ftn_int isrcproc, ftn_int nprocs)
        {
            return FORTRAN(numroc)(&n, &nb, &iproc, &isrcproc, &nprocs); 
        }
        #endif
};


template <processing_unit_t pu>
class linalg;

template<> 
class linalg<CPU>: public linalg_base
{
    public:
        
        /// Compute C = alpha * op(A) * op(B) + beta * op(C) with raw pointers.
        template <typename T>
        static void gemm(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, T alpha, T* A, ftn_int lda,
                         T* B, ftn_int ldb, T beta, T* C, ftn_int ldc);
        
        /// Compute C = op(A) * op(B) operation with raw pointers.
        template <typename T>
        static void gemm(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, T* A, ftn_int lda, T* B, ftn_int ldb, 
                         T* C, ftn_int ldc);

        /// Compute C = alpha * op(A) * op(B) + beta * op(C) with matrix objects.
        template <typename T>
        static void gemm(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, T alpha, matrix<T>& A, matrix<T>& B,
                         T beta, matrix<T>& C);

        /// Compute C = op(A) * op(B) operation with matrix objects.
        template <typename T>
        static void gemm(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, matrix<T>& A, matrix<T>& B,
                         matrix<T>& C);
                         
        ///template<typename T>
        ///static void hemm(int side, int uplo, int32_t m, int32_t n, T alpha, T* a, int32_t lda, 
        ///                 T* b, int32_t ldb, T beta, T* c, int32_t ldc);

        ///template<typename T>
        ///static void gemv(int trans, int32_t m, int32_t n, T alpha, T* a, int32_t lda, T* x, int32_t incx, 
        ///                 T beta, T* y, int32_t incy);

        ////// generic interface to p?gemm
        ///template <typename T>
        ///static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T alpha, 
        ///                 dmatrix<T>& a, int32_t ia, int32_t ja, dmatrix<T>& b, int32_t ib, int32_t jb, T beta, 
        ///                 dmatrix<T>& c, int32_t ic, int32_t jc);

        ////// simple interface to p?gemm: all matrices start form (0, 0) corner
        ///template <typename T>
        ///static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
        ///                 T alpha, dmatrix<T>& a, dmatrix<T>& b, T beta, dmatrix<T>& c);



        template <typename T>
        static ftn_int getrf(ftn_int m, ftn_int n, T* A, ftn_int lda, ftn_int* ipiv);
        
        template <typename T>
        static ftn_int getri(ftn_int n, T* A, ftn_int lda, ftn_int* ipiv);
        
        /// Invert a general matrix.
        template <typename T>
        void geinv(ftn_int n, matrix<T>& A);

        template <typename T>
        static ftn_int getrf(ftn_int m, ftn_int n, dmatrix<T>& A, ftn_int ia, ftn_int ja, ftn_int* ipiv);

        template <typename T>
        static int getri(ftn_int n, dmatrix<T>& A, ftn_int ia, ftn_int ja, ftn_int* ipiv);

        /// Invert a general distributed matrix.
        template <typename T>
        void geinv(ftn_int n, dmatrix<T>& A);
};







#endif // __LINALG_H__

